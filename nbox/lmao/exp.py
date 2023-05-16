"""
NimbleBox LMAO is our general purpose observability tool for any kind of computation you might have.
"""

import os
from json import dumps, loads
from functools import lru_cache
from typing import Dict, Any, List, Optional, Union

import nbox.utils as U
from nbox.utils import logger, SimplerTimes
from nbox.auth import secret, AuthConfig, JobDetails
from nbox.relics import Relics

# all the sublime -> hyperloop stuff
from nbox.lmao.lmao_rpc_client import (
  AgentDetails,
  RunLog,
  Run as RunProto,
  InitRunRequest
)
from nbox.observability.system import SystemMetricsLogger
from nbox.lmao.common import get_git_details, get_lmao_stub, get_record, get_project



"""
Client library that the user will use to interact with the LMAO server.
"""

class _lmaoConfig:
  # _lmaoConfig.kv contains all the objects that the class LMAO needs to work correctly, however
  # we will also need to take care of the things we want to show 
  kv = {}
  def set(
    project_id: str,
    experiment_id: str,
    save_to_relic: bool,
    enable_system_monitoring: bool,
    store_git_details: bool,
  ) -> None:
    _lmaoConfig.kv = {
      "project_id": project_id,
      "experiment_id": experiment_id,
      "save_to_relic": save_to_relic,
      "enable_system_monitoring": enable_system_monitoring,
      "store_git_details": store_git_details,
    }


class LmaoTypes:
  EXP = "experiment"
  LIVE = "live"

  def all() -> List[str]:
    return [LmaoTypes.EXP, LmaoTypes.LIVE]


class Lmao():
  def __init__(
    self,
    project_id: Optional[str] = "",
    experiment_id: Optional[str] = "",
    metadata: Dict[str, Any] = {},
    save_to_relic: bool = True,
    enable_system_monitoring: bool = False,
    store_git_details: bool = True,
  ) -> None:
    """`Lmao` is the client library for using NimbleBox Monitoring. It talks to your monitoring instance running on your build
    and stores the information in the `project_name` or `project_id`. This object inherently doesn't care what you are actually
    logging and rather concerns itself with ensuring storage.

    **Note**: All arguments are optional, if the `_lmaoConfig.kv` is set.

    Args:
      project_name (str, optional): The name of the project. Defaults to "".
      project_id (str, optional): The id of the project. Defaults to "".
      experiment_id (str, optional): The id of the experiment. Defaults to "".
      metadata (Dict[str, Any], optional): Any metadata that you want to store. Defaults to {}.
      save_to_relic (bool, optional): Whether to save the data to the relic. Defaults to False.
      enable_system_monitoring (bool, optional): Whether to enable system monitoring. Defaults to False.
      store_git_details (bool, optional): Whether to store git details. Defaults to True.
    """
    if _lmaoConfig.kv:
      # load all the values from the config
      project_id = _lmaoConfig.kv["project_id"]
      experiment_id = _lmaoConfig.kv["experiment_id"]
      save_to_relic = _lmaoConfig.kv["save_to_relic"]
      enable_system_monitoring = _lmaoConfig.kv["enable_system_monitoring"]
      store_git_details = _lmaoConfig.kv["store_git_details"]

    self.project_id = project_id
    self.experiment_id = experiment_id
    self.save_to_relic = save_to_relic
    self.enable_system_monitoring = enable_system_monitoring
    self.store_git_details = store_git_details
    self.workspace_id = secret.workspace_id

    self.agent = secret.get_agent_details()

    # now set the supporting keys
    self._total_logged_elements = 0
    self.completed = False
    self.relic: Relics = None
    self.sys_montior: SystemMetricsLogger = None

    # create connection and initialise the run
    self.lmao = get_lmao_stub()
    self.project = get_project(project_id = self.project_id)
    if self.project is None:
      raise Exception(f"Project with id {self.project_id} does not exist")

    self.config = self._get_config(metadata = metadata)
    self.run: RunProto = self._init_experiment(
      project_id = project_id,
      config = self.config,
      experiment_id = experiment_id
    )

    if self.save_to_relic:
      self.relic = Relics(id = self.project.relic_id, prefix = self.run.save_location)
      logger.info(f"Will store everything in relic '{self.relic}'")

    # system metrics monitoring, by default is enabled optionally turn it off
    if self.enable_system_monitoring:
      raise NotImplementedError
      self.sys_montior = SystemMetricsLogger(self)
      self.sys_montior.start()

  def __repr__(self) -> str:
    return f"Lmao({self.project_id}, {self.experiment_id}, {self._total_logged_elements})"

  def __del__(self):
    if self.sys_montior is not None:
      self.sys_montior.stop()

  def _get_config(self, metadata: Dict[str, Any]):
    # this is the config value that is used to store data on the plaform, user cannot be allowed to have
    # like a full access to config values
    log_config: Dict[str, Any] = {
      "run_kwargs": metadata
    }

    # check if the current folder from where this code is being executed has a .git folder
    # NOTE: in case of NBX-Jobs the current folder ("./") is expected to contain git by default
    log_config["git"] = None
    if os.path.exists(".git") and self.store_git_details:
      log_config["git"] = get_git_details("./")
    return log_config

  def _init_experiment(self, project_id, config: Dict[str, Any] = {}, experiment_id: str = "") -> RunProto:
    # update the server or create new experiment
    agent_details = AgentDetails(
      workspace_id = self.workspace_id,
      nbx_job_id = "local",
      nbx_run_id = U.SimplerTimes.get_now_str(),
    )
    if type(self.agent) == JobDetails:
      a: JobDetails = self.agent
      agent_details.type = AgentDetails.NBX.JOB
      agent_details.nbx_job_id = a.job_id
      agent_details.nbx_run_id = a.run_id

    if experiment_id:
      action = "Updated"
      run_details = self.lmao.get_run_details(RunProto(
        workspace_id = self.workspace_id,
        project_id = project_id,
        experiment_id = experiment_id,
      ))
      if not run_details:
        # TODO: Make a custom exception of this
        raise Exception("Server Side exception has occurred, Check the log for details")
      if run_details.experiment_id:
        # means that this run already exists so we need to make an update call
        ack = self.lmao.update_run_status(RunProto(
          workspace_id = self.workspace_id,
          project_id = project_id,
          experiment_id = run_details.experiment_id,
          agent = agent_details,
          update_keys = ["agent"],
        ))
        if not ack.success:
          raise Exception(f"Failed to update run status! {ack.message}")
    else:
      action = "Created"
      run_details = self.lmao.init_run(
        InitRunRequest(
          workspace_id = self.workspace_id,
          project_id = project_id,
          agent_details=agent_details,
          config = dumps(config),
        )
      )

    logger.info(
      f"{action} experiment tracker\n"
      f" project: {project_id}\n"
      f"      id: {run_details.experiment_id}\n"
      f"    link: {secret.nbx_url}/workspace/{self.workspace_id}/projects/{project_id}#Experiments\n"
    )

    return run_details

  """The functions below are the ones supposed to be used."""

  @property
  def run_config(self) -> Dict[str, Any]:
    return loads(self.run.config)

  @lru_cache(maxsize=1)
  def get_relic(self):
    """Get the underlying Relic for more advanced usage patterns."""
    if self.save_to_relic:
      return self.relic
    raise Exception("Relic is not enabled for this run, set save_to_relic = True")

  def log(self, y: Dict[str, Union[int, float, str]], step = None, *, log_type: str = RunLog.LogType.USER):
    """Log a single level dictionary to the platform at any given step. This function does not really care about the
    information that is being logged, it just logs it to the platform."""
    if self.completed:
      raise Exception("Run already completed, cannot log more data!")

    step = step or self._total_logged_elements
    if step < 0:
      raise Exception("Step must be >= 0")
    run_log = RunLog(
      workspace_id = self.workspace_id,
      project_id=self.project_id,
      experiment_id = self.run.experiment_id,
      log_type=log_type
    )
    for k,v in y.items():
      # TODO: @yashbonde replace Record with RecordColumn
      record = get_record(k, v)
      record.step = step
      run_log.data.append(record)

    ack = self.lmao.on_log(run_log)
    if not ack.success:
      logger.error(f"  >> Server Error\n{ack.message}")
      raise Exception("Server Error")

    self._total_logged_elements += 1

  def end(self):
    """End the run to declare it complete. This is more of a convinience function than anything else. For example when you
    are monitoring a live API you may never know when the experiment is complete. This function locks the experiment name
    so it can't be modified."""
    if self.completed:
      logger.error("Run already completed, cannot end it again!")
      return None

    logger.info("Ending run")
    ack = self.lmao.on_train_end(self.run)
    if not ack.success:
      logger.error("  >> Server Error")
      for l in ack.message.splitlines():
        logger.error("  " + l)
      raise Exception("Server Error")
    self.completed = True
    if self.enable_system_monitoring:
      self.sys_montior.stop()

  def save_file(self, *files: List[str]):
    logger.info(f"Saving files: {files}")
    return self.add_files(*files)

  def add_files(self, *files: List[str]):
    """
    Register a file save. If `save_to_relics` is not set, this function is a no-op.

    Args:
      files: The list of files to save. This can be a list of files or a list of folders. If a folder is passed, all the files in the folder will be uploaded.
    """
    # manage all the complexity of getting the list of RelicFile
    all_files = []
    for folder_or_file in files:
      if os.path.isfile(folder_or_file):
        all_files.append(folder_or_file)
      elif os.path.isdir(folder_or_file):
        all_files.extend(U.get_files_in_folder(folder_or_file, abs_path=False))
      else:
        raise Exception(f"File or Folder not found: {folder_or_file}")

    # when we add something cool we can use this
    logger.debug(f"Storing {len(all_files)} files")
    if self.save_to_relic:
      relic = self.get_relic()
      for f in all_files:
        relic.put(f)
    return all_files
