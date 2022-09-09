"""
NimbleBox LMAO is our general purpose observability tool for any kind of computation you might have.
"""

import os
import re
from git import Repo
from json import dumps
from requests import Session
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timezone

import nbox.utils as U
from nbox import Instance
from nbox.utils import logger
from nbox.auth import secret
from nbox.nbxlib.tracer import Tracer
from nbox.relics import RelicsNBX
from nbox.sublime.lmao_client import (
  LMAO_Stub, Record, File, FileList, AgentDetails, RunLog, Run, InitRunRequest, ListProjectsRequest, RelicFile
)
from nbox.observability.system import SystemMetricsLogger

DEBUG_LOG_EVERY = 100
INFO_LOG_EVERY = 100


class Lmao():
  def __init__(
    self,
    workspace_id: str,
    metadata: Dict[str, Any],
    project_name: Optional[str] = "",
    project_id: Optional[str] = "",
  ) -> None:
    """``Lmao`` is the client library for using NimbleBox Monitoring. It talks to your monitoring instance running on your build
    and stores the information in the ``project_name`` or ``project_id``. This object inherently doesn't care what you are actually
    logging and rather concerns itself with ensuring storage."""
    self.workspace_id = workspace_id

    self.nbx_job_folder = U.env.NBOX_JOB_FOLDER("")
    self._total_logged_elements = 0 # this variable keeps track for logging
    self.completed = False
    self.tracer = None
    self.username = secret.get("username")
    self.relic: RelicsNBX = None
    self._nbx_run_id = None
    self._nbx_job_id = None

    # create connection and initialise the run
    self._create_connection(workspace_id)
    self._init(project_name=project_name, project_id=project_id, config=metadata)

  def _create_connection(self, workspace_id: str):
    # prepare the URL
    # id_or_name, workspace_id = U.split_iw(instance_id)
    id_or_name = f"monitoring-{workspace_id}"
    logger.info(f"id_or_name: {id_or_name}")
    logger.info(f"workspace_id: {workspace_id}")
    instance = Instance(id_or_name, workspace_id)
    try:
      open_data = instance.open_data
    except AttributeError:
      raise Exception(f"Is instance '{instance.project_id}' running?")
    build = "build"
    if "app.c." in secret.get("nbx_url"):
      build = "build.c"
    url = f"https://server-{open_data['url']}.{build}.nimblebox.ai/"
    logger.info(f"URL: {url}")

    # create a tracer object that will load all the information
    self.tracer = Tracer(start_heartbeat=False)
    self._nbx_run_id = self.tracer.run_id
    self._nbx_job_id = self.tracer.job_id
    if self._nbx_run_id is not None:
      # update the username you have
      secret.put("username", self.tracer.job_proto.auth_info.username)
      self.username = secret.get("username")

    # create a session with the auth header
    _session = Session()
    _session.headers.update({
      "NBX-TOKEN": open_data["token"],
      "X-NBX-USERNAME": self.username,
    })

    # define the stub
    # self.lmao = LMAO_Stub(url = "http://127.0.0.1:8080", session = Session()) # debug
    self.lmao = LMAO_Stub(url = url, session = _session)

  def _get_name_id(self, project_id: str = None, project_name: str = None):
    all_projects = self.lmao.list_projects(
      _ListProjectsRequest = ListProjectsRequest(
        workspace_id = self.workspace_id,
        project_id_or_name = project_id if project_id else project_name,
      )
    )
    if not all_projects.projects:
      # means we have to initialise this project first and then we will pull the details after the init_run
      if project_id:
        raise Exception(f"Project with id {project_id} not found, please create a new one with project_name")
      logger.info(f"Project '{project_name}' not found, will create a new one.")
    
    if project_id:
      common = list(filter(lambda x: x.project_id == project_name, all_projects.projects))
      if len(common) > 1:
        logger.error(f"Multiple entries for {project_id} found, something went wrong from our side.")
        raise Exception(f"Database duplicate entry for project_id: {project_id}")
      elif len(common) == 1:
        project_name = common[0].project_name
        project_id = common[0].project_id
    else:
      common = list(filter(lambda x: x.project_name == project_name, all_projects.projects))
      if len(common) > 1:
        logger.error(f"Project '{project_name}' found multiple times, please use project_id instead.")
        raise Exception("Ambiguous project name, please use project_id instead.")
      elif len(common) == 1:
        project_name = common[0].project_name
        project_id = common[0].project_id
    return project_id, project_name

  def _init(self, project_name: Optional[str] = "", config: Dict[str, Any] = {}, project_id: Optional[str] = ""):
    # do a quick lookup and see if the project exists, if not, create it
    if project_id:
      project_id, project_name = self._get_name_id(project_id=project_id)
      if not project_id:
        raise Exception(f"Project with id {project_id} not found, please create a new one with project_name")
    else:
      project_id, project_name = self._get_name_id(project_name=project_name)
      if not project_id:
        logger.info(f"Project '{project_name}' not found, will create a new one.")

    # this is the config value that is used to store data on the plaform, user cannot be allowed to have
    # like a full access to config values
    log_config: Dict[str, Any] = {
      "user_config": config
    }

    # check if the current folder from where this code is being executed has a .git folder
    # NOTE: in case of NBX-Jobs the current folder ("./") is expected to contain git by default
    log_config["git"] = None
    if os.path.exists(".git"):
      log_config["git"] = get_git_details("./")

    # continue as before
    self._agent_details = AgentDetails(
      nbx_job_id = self._nbx_job_id or "jj_guvernr",
      nbx_run_id = self._nbx_run_id or "fake_run",
    )
    run_details = self.lmao.init_run(
      _InitRunRequest = InitRunRequest(
        agent_details=self._agent_details,
        created_at = get_timestamp(),
        project_name = project_name,
        project_id = project_id,
        config = dumps(log_config),
      )
    )

    if not run_details:
      # TODO: Make a custom exception of this
      raise Exception("Server Side exception has occurred, Check the log for details")

    if not project_id:
      project_id, project_name = self._get_name_id(project_name = project_name)

    self.project_name = project_name
    self.project_id = project_id
    self.config = config

    # TODO: change f-string here when refactoring `run_id` to `exp_id`
    # logger.info(f"Assigned run_id: {run_details.run_id}")
    self.run = run_details
    logger.info(f"Created a new LMAO run")
    logger.info(f"    id: {run_details.run_id}")
    logger.info(f"  link: https://app.nimblebox.ai/workspace/{self.workspace_id}/monitoring/{self.project_id}/{run_details.run_id}")

    # now initialize the relic
    if not U.env.NBOX_LMAO_DISABLE_RELICS():
      # The relic will be the project id
      self.relic = RelicsNBX("experiments", self.workspace_id, create = True)
      logger.info(f"Will store everything in folder: {self.experiment_prefix}")

    # system metrics monitoring, by default is enabled optionally turn it off
    if not U.env.NBOX_LMAO_DISABLE_SYSTEM_METRICS():
      self.system_monitoring = SystemMetricsLogger(self)
      self.system_monitoring.start()

  @property
  def experiment_prefix(self):
    prefix = f"{self.project_id}/{self.run.run_id}"
    if self._nbx_job_id is not None:
      prefix += f"_{self._nbx_job_id}@{self._nbx_run_id}"
    prefix += "/"
    return prefix

  """The functions below are the ones supposed to be used."""

  def log(self, y: Dict[str, Union[int, float, str]], step = None, *, log_type: str = RunLog.LogType.USER):
    """Log a single level dictionary to the platform at any given step. This function does not really care about the
    information that is being logged, it just logs it to the platform."""
    if self.completed:
      raise Exception("Run already completed, cannot log more data!")

    if self._total_logged_elements % DEBUG_LOG_EVERY == 0:
      logger.debug(f"Logging: {y.keys()} | {self._total_logged_elements}")

    if self._total_logged_elements % INFO_LOG_EVERY == 0:
      logger.info(f"Logging: {y.keys()} | {self._total_logged_elements}")

    step = step if step is not None else get_timestamp()
    if step < 0:
      raise Exception("Step must be <= 0")
    run_log = RunLog(run_id = self.run.run_id, log_type=log_type)
    for k,v in y.items():
      record = get_record(k, v)
      record.step = step
      run_log.data.append(record)

    ack = self.lmao.on_log(_RunLog = run_log)
    if not ack.success:
      logger.error("  >> Server Error")
      for l in ack.message.splitlines():
        logger.error("  " + l)
      raise Exception("Server Error")

    self._total_logged_elements += 1

  def save_file(self, *files: List[str]):
    """
    Register a file save and upload it by talking to the NimbleBox Relics Backend. User should be aware of some structures
    that we follow for standardizing the data. All the experiments are going to be tracked under the following pattern:

    #. ``relic_name`` is going to be the experiment ID, so any changes to the name will not affect relic storage
    #. ``{experiment_id}(_{job_id}@{run_id})`` is the name of the folder which contains all the artifacts in the experiment.

    If relics is not enabled, this function will simply log to the LMAO DB.

    dk.save_file("foo.t", "/bar/", "baz.t", "/bar/roo/")
    """
    logger.info(f"Saving files: {files}")

    # manage all the complexity of getting the list of RelicFile
    all_files = []
    for folder_or_file in files:
      if os.path.isfile(folder_or_file):
        all_files.append(folder_or_file)
      elif os.path.isdir(folder_or_file):
        all_files.extend(U.get_files_in_folder(folder_or_file, self.username, self.workspace_id))
      else:
        raise Exception(f"File or Folder not found: {folder_or_file}")

    logger.debug(f"Storing {len(all_files)} files")
    if self.relic is not None:
      logger.info(f"Uploading files to relic: {self.relic}")
      for f in all_files:
        print(f, f"{self.experiment_prefix}{f.strip('/')}")
        self.relic.put_to(f, f"{self.experiment_prefix}{f.strip('/')}")

    # log the files in the LMAO DB for sanity
    fl = FileList(run_id = self.run.run_id)
    fl.files.extend([File(relic_file = RelicFile(name = x)) for x in all_files])
    self.lmao.on_save(_FileList = fl)

  def end(self):
    """End thr run to declare it complete. This is more of a convinience function than anything else. For example when you
    are monitoring a live API you may never know when the experiment is complete. This function locks the experiment name
    so it can't be modified."""
    if self.completed:
      logger.error("Run already completed, cannot end it again!")
      return None

    logger.info("Ending run")
    ack = self.lmao.on_train_end(_Run = Run(run_id=self.run.run_id,))
    if not ack.success:
      logger.error("  >> Server Error")
      for l in ack.message.splitlines():
        logger.error("  " + l)
      raise Exception("Server Error")
    self.completed = True
    self.system_monitoring.stop()

"""
Utility functions below.
"""

def get_record(k: str, v: Union[int, float, str]) -> Record:
  """Function to create a Record protobuf object from a key and value."""
  _tv = type(v)
  assert _tv in [int, float, str], f"[key = {k}] '{_tv}' is not a valid type"
  _vt = {
    int: Record.DataType.INTEGER,
    float: Record.DataType.FLOAT,
    str: Record.DataType.STRING,
  }[_tv]
  record = Record(key = k, value_type = _vt)
  if _tv == int:
    record.integer_data.append(v)
  elif _tv == float:
    record.float_data.append(v)
  elif _tv == str:
    record.string_data.append(v)
  return record

def get_git_details(folder):
  """If there is a .git folder in the folder, return some details for that."""
  repo = Repo(folder)

  # check for any unstaged files
  uncommited_files = {}
  diff = repo.index.diff(None)
  for f in diff:
    path = f.a_path or f.b_path # when new file is added, a_path is None
    uncommited_files[path] = f.change_type
  if uncommited_files:
    logger.warning(f"Uncommited files: {uncommited_files}")

  # get the remote url
  try:
    remote_url = repo.remote().url

    # clean the remote_url because it can sometimes contain the repo token as well.
    # this can become security hazard. so if you have an example, that is not suppported:
    #   go ahead, make a PR!
    if "github.com" in remote_url:
      remote_url = re.sub(r"ghp_\w+@", "", remote_url)
  except ValueError:
    remote_url = None

  # get the size of the repository
  size = None
  for line in repo.git.count_objects("-v").splitlines():
    if line.startswith("size:"):
      size = int(line[len("size:") :].strip())
  if size > (1 << 30):
    logger.warning(f"Repository size over 1GB, you might want to work on it")

  return {
    "remote_url": remote_url,
    "branch": repo.active_branch.name,
    "commit": repo.head.commit.hexsha,
    "uncommited_files": uncommited_files,
    "untracked_files": repo.untracked_files,
    "size": size,
  }

def get_timestamp():
  """Get the current timestamp."""
  return int(datetime.now(timezone.utc).timestamp())
