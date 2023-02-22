"""
NimbleBox LMAO is our general purpose observability tool for any kind of computation you might have.
"""

# drift detection and all
# run.log_dataset(
#     dataset_name='train',
#     features=X_train,
#     predictions=y_pred_train,
#     actuals=y_train,
# )
# run.log_dataset(
#     dataset_name='test',
#     features=X_test,
#     predictions=y_pred_test,
#     actuals=y_test,
# )

import os
import re
import sys
import shlex
import zipfile
from git import Repo
from json import dumps, loads
from functools import lru_cache
from requests import Session
from typing import Dict, Any, List, Optional, Union
from subprocess import Popen
from google.protobuf.field_mask_pb2 import FieldMask

import nbox.utils as U
from nbox import Instance
from nbox.utils import logger, SimplerTimes
from nbox.auth import secret, AuthConfig, auth_info_pb, JobDetails
from nbox.nbxlib.tracer import Tracer
from nbox.relics import Relics
from nbox.jobs import Job, upload_job_folder
from nbox.hyperloop.common.common_pb2 import Resource
from nbox.init import nbox_grpc_stub, nbox_ws_v1
from nbox.messages import message_to_dict
from nbox.hyperloop.jobs.nbox_ws_pb2 import UpdateJobRequest, JobRequest
from nbox.hyperloop.jobs.job_pb2 import Job as JobProto

# all the sublime -> hyperloop stuff
from nbox.sublime.lmao_rpc_client import (
  LMAO_Stub, # main stub class
  Record, AgentDetails, RunLog,
  Run, InitRunRequest, ListProjectsRequest,
)
from nbox.observability.system import SystemMetricsLogger


"""
functional components of LMAO
"""

def get_lmao_stub() -> LMAO_Stub:
  lmao_stub = LMAO_Stub(url = secret(AuthConfig.url) + "/monitoring", session = nbox_ws_v1._session)
  return lmao_stub

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
  """If there is a `.git` folder in the folder, return some details for that."""
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

"""
Client library that the user will use to interact with the LMAO server.
"""

class _lmaoConfig:
  # _lmaoConfig.kv contains all the objects that the class LMAO needs to work correctly, however
  # we will also need to take care of the things we want to show 
  kv = {}
  def set(
    project_name: str,
    project_id: str,
    experiment_id: str,
    save_to_relic: bool,
    enable_system_monitoring: bool,
    store_git_details: bool,
  ) -> None:
    _lmaoConfig.kv = {
      "project_name": project_name,
      "project_id": project_id,
      "experiment_id": experiment_id,
      "save_to_relic": save_to_relic,
      "enable_system_monitoring": enable_system_monitoring,
      "store_git_details": store_git_details,
    }


class Lmao():
  def __init__(
    self,
    project_name: Optional[str] = "",
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
      # project_name = _lmaoConfig.kv["project_name"]
      project_id = _lmaoConfig.kv["project_id"]
      experiment_id = _lmaoConfig.kv["experiment_id"]
      save_to_relic = _lmaoConfig.kv["save_to_relic"]
      enable_system_monitoring = _lmaoConfig.kv["enable_system_monitoring"]
      store_git_details = _lmaoConfig.kv["store_git_details"]

    # self.project_name = project_name
    self.project_id = project_id
    self.experiment_id = experiment_id
    self.save_to_relic = save_to_relic
    self.enable_system_monitoring = enable_system_monitoring
    self.store_git_details = store_git_details
    self.workspace_id = secret(AuthConfig.workspace_id)

    self.agent = secret.get_agent_details()

    # now set the supporting keys
    self.nbx_job_folder = U.env.NBOX_JOB_FOLDER("")
    self._total_logged_elements = 0 # this variable keeps track for logging
    self.completed = False
    self.relic: Relics = None
    self.system_monitoring: SystemMetricsLogger = None

    # create connection and initialise the run
    self._init(project_id, config = metadata)

  def __del__(self):
    if self.system_monitoring is not None:
      self.system_monitoring.stop()

  def _init(self, project_id, config: Dict[str, Any] = {}):
    # create a tracer object that will load all the information
    self.lmao = get_lmao_stub()

    # this is the config value that is used to store data on the plaform, user cannot be allowed to have
    # like a full access to config values
    log_config: Dict[str, Any] = {
      "user_config": config
    }

    # check if the current folder from where this code is being executed has a .git folder
    # NOTE: in case of NBX-Jobs the current folder ("./") is expected to contain git by default
    log_config["git"] = None
    if os.path.exists(".git") and self.store_git_details:
      log_config["git"] = get_git_details("./")

    # update the server or create new experiment
    self._agent_details = AgentDetails(workspace_id = self.workspace_id)
    if type(self.agent) == JobDetails:
      a: JobDetails = self.agent
      self._agent_details.type = AgentDetails.NBX.JOB
      self._agent_details.nbx_job_id = a.job_id
      self._agent_details.nbx_run_id = a.run_id

    if self.experiment_id:
      action = "Updated"
      run_details = self.lmao.get_run_details(Run(
        workspace_id = self.workspace_id,
        project_id = project_id,
        experiment_id = self.experiment_id,
      ))
      if not run_details:
        # TODO: Make a custom exception of this
        raise Exception("Server Side exception has occurred, Check the log for details")
      if run_details.experiment_id:
        # means that this run already exists so we need to make an update call
        ack = self.lmao.update_run_status(Run(
          workspace_id = self.workspace_id,
          project_id = project_id,
          experiment_id = run_details.experiment_id,
          agent = self._agent_details,
          update_keys = ["agent"],
        ))
        if not ack.success:
          raise Exception(f"Failed to update run status! {ack.message}")
    else:
      action = "Created"
      run_details = self.lmao.init_run(
        InitRunRequest(
          workspace_id = self.workspace_id,
          agent_details=self._agent_details,
          project_id = project_id,
          config = dumps(log_config),
        )
      )

    self.project_id = run_details.project_id
    self.run = run_details

    logger.info(
      f"{action} LMAO run\n"
      f" project: {self.project_id}\n"
      f"      id: {self.run.experiment_id}\n"
      f"    link: https://app.nimblebox.ai/workspace/{self.workspace_id}/monitoring/{self.project_id}/{self.run.experiment_id}\n"
    )

    # now initialize the relic
    if self.save_to_relic:
      out = self.run.save_location.split(":")
      if len(out) != 2:
        logger.error(f"Invalid save location, this is an error from NBX side.\n  Fix: Create a new project and try again.")
        raise Exception(f"cant find correct relic ID: {self.run.save_location}")
      relic_id, self.experiment_prefix = out
      self.relic = Relics(relic_id)
      logger.info(f"Will store everything in relic '{relic_id}' folder: {self.experiment_prefix}")

    # system metrics monitoring, by default is enabled optionally turn it off
    if self.enable_system_monitoring:
      self.system_monitoring = SystemMetricsLogger(self)
      self.system_monitoring.start()

  """The functions below are the ones supposed to be used."""

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

    step = step if step is not None else SimplerTimes.get_now_i64()
    if step < 0:
      raise Exception("Step must be <= 0")
    run_log = RunLog(
      workspace_id = self.workspace_id,
      project_id=self.project_id,
      experiment_id = self.run.experiment_id,
      log_type=log_type
    )
    for k,v in y.items():
      # TODO:@yashbonde replace Record with RecordColumn
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
      self.system_monitoring.stop()

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
        all_files.extend(U.get_files_in_folder(folder_or_file))
      else:
        raise Exception(f"File or Folder not found: {folder_or_file}")

    # when we add something cool we can use this
    logger.debug(f"Storing {len(all_files)} files")
    if self.save_to_relic:
      relic = self.get_relic()
      logger.info(f"Uploading files to relic: {relic}")
      for f in all_files:
        relic.put(f)

class __LmaoBundle:
  # files = Dict[str, Any]
  def __init__(self, name: str, meta: Dict[str, Any]):
    # bundle = LmaoBundle("bundle_name")
    # bundle = LmaoBundle("bundle_name", {"key": "value"})
    pass

  def add_files() -> None:
    # bundle.add_files("model.pt")
    # bundle.add_files("model/*.pt")
    # bundle.add_files("model/config.json", "model/tokenizer.json")
    pass

  def upload() -> None:
    # bundle.upload()
    pass

  def download() -> str:
    # bundle.download() -> "nbx_bundle_{id}" directory downloaded
    # bundle.download("model.pt") -> "nbx_bundle_{id}/model.pt"
    # bundle.download(to = "this_folder/") -> "this_folder"
    # bundle.download("model.pt", to = "this_folder/") -> "this_folder/model.pt"
    pass

  def trail() -> List[str]:
    # return a list of all the modifier of artifacts
    pass

  def verify() -> bool:
    # bundle.verify("local_folder/") -> True
    # bundle.verify("modified_local_folder/") -> False
    pass

"""
For some Experiences you want to have CLI control so this class manages that

```
lmao run fp:fn project_name_or_id
```
"""

class ExperimentConfig:
  def __init__(
    self,
    run_kwargs: Dict[str, Any],
    git: Dict[str, Any],
    resource: Resource,
    cli_comm: str,
    save_to_relic: bool = True,
    enable_system_monitoring: bool = False,
  ):
    """In an ideal world this would be a protobuf message, but we are not there yet. This contains all the
    things that are stored in the DB and FE can use this to render elements in the UI.

    Args:
      run_kwargs (Dict[str, Any]): All the arguments that the user passed to the `nbx lmao run ...` CLI
      git (Dict[str, Any]): details for the git repo
      resource (Resource): Resource pb object that contains the details of the resource
      cli_comm (str): The CLI command that was used to run the experiment, user can use this to reproduce the experiment
      save_to_relic (bool): If the user wants to save the experiment to the relic by default
      enable_system_monitoring (bool): If the user wants to enable system monitoring
    """
    self.run_kwargs = run_kwargs
    self.git = git
    self.resource = resource
    self.cli_comm = cli_comm
    self.save_to_relic = save_to_relic
    self.enable_system_monitoring = enable_system_monitoring

  def to_dict(self):
    return {
      "run_kwargs": self.run_kwargs,
      "git": self.git,
      "resource": message_to_dict(self.resource),
      "cli_comm": self.cli_comm,
      "save_to_relic": self.save_to_relic,
      "enable_system_monitoring": self.enable_system_monitoring,
    }

  def to_json(self):
    return dumps(self.to_dict())

  @classmethod
  def from_json(cls, json_str):
    d = loads(json_str)
    print(d["resource"])
    d["resource"] = Resource(
      cpu = str(d["resource"]["cpu"]),
      memory = str(d["resource"]["memory"]),
      disk_size = str(d["resource"]["disk_size"]),
      gpu = str(d["resource"]["gpu"]),
      gpu_count = str(d["resource"]["gpu_count"]),
      timeout = int(d["resource"]["timeout"]),
      max_retries = int(d["resource"]["max_retries"]),
    )
    return cls(**d)


@lru_cache()
def get_project_name_id(project_name_or_id: str, workspace_id: str):
  lmao_stub = get_lmao_stub()
  out = lmao_stub.list_projects(ListProjectsRequest(workspace_id=workspace_id, project_id_or_name=project_name_or_id))
  if not out.projects:
    logger.error(f"Project: {project_name_or_id} not found")
    logger.error(f"Will automatically create one")
    name = project_name_or_id
    id = None
  elif len(out.projects) > 1:
    raise ValueError(f"Multiple projects found for: {project_name_or_id}")
  else:
    p = out.projects[0]
    name = p.project_name
    id = p.project_id
  return name, id


class LmaoCLI:
  # This class will only manage the things that need to talk to the DB and everything else is offloaded
  def upload(self, **kwargs):
    raise ValueError("`nbx lmao upload` is not changed to `nbx lmao run`")

  def run(
    self,
    init_path: str,
    project_name_or_id: str,
    trigger: bool = False,

    # all the arguments for the git thing
    untracked: bool = False,
    untracked_no_limit: bool = False,

    # all the things for resources
    resource_cpu: str = "100m",
    resource_memory: str = "128Mi",
    resource_disk_size: str = "1Gi",
    resource_gpu: str = "none",
    resource_gpu_count: str = "0",
    resource_timeout: int = 120_000,
    resource_max_retries: int = 2,

    # the following arguments are used for the initialisation of lmao class
    save_to_relic: bool = True,
    enable_system_monitoring: bool = False,

    # the following things are needed for the different modules in the process
    relics_kwargs: Dict[str, Any] = {},

    # any other things to pass to the function / class being called
    **run_kwargs
  ):
    """Upload and register a new run for a NBX-LMAO project.

    Args:
      init_path (str): This can be a path to a `folder` or can be optionally of the structure `fp:fn` where `fp` is the path to the file and `fn` is the function name.
      project_name_or_id (str): The name or id of the LMAO project.
      trigger (bool, optional): Defaults to False. If True, will trigger the run after uploading.
      untracked (bool, optional): If True, then untracked files below 1MB will also be zipped and uploaded. Defaults to False.
      untracked_no_limit (bool, optional): If True, then all untracked files will also be zipped and uploaded. Defaults to False.
      resource_cpu (str, optional): Defaults to "100m". The CPU resource to allocate to the run.
      resource_memory (str, optional): Defaults to "128Mi". The memory resource to allocate to the run.
      resource_disk_size (str, optional): Defaults to "1Gi". The disk size resource to allocate to the run.
      resource_gpu (str, optional): Defaults to "none". The GPU resource to use.
      resource_gpu_count (str, optional): Defaults to "0". Number of GPUs allocated to the experiment.
      resource_timeout (int, optional): Defaults to 120_000. The timeout between two consecutive runs, honoured but not guaranteed.
      resource_max_retries (int, optional): Defaults to 2. The maximum number of retries for a run.
      save_to_relic (bool, optional): Defaults to True. If True, will save the files to a relic when lmao.save_file() is called.
      enable_system_monitoring (bool, optional): Defaults to False. If True, will enable system monitoring.
      **run_kwargs: These are the kwargs that will be passed to your function.
    """
    workspace_id = secret(AuthConfig.workspace_id)

    # fix data type conversion caused by the CLI
    resource_cpu = str(resource_cpu)
    resource_memory = str(resource_memory)
    resource_disk_size = str(resource_disk_size)
    resource_gpu = str(resource_gpu)
    resource_gpu_count = str(resource_gpu_count)
    resource_timeout = int(resource_timeout)
    resource_max_retries = int(resource_max_retries)

    # reconstruct the entire CLI command so we can show it in the UI
    reconstructed_cli_comm = (
      f"nbx lmao upload '{init_path}' '{project_name_or_id}'"
      f" --resource_cpu '{resource_cpu}'"
      f" --resource_memory '{resource_memory}'"
      f" --resource_disk_size '{resource_disk_size}'"
      f" --resource_gpu '{resource_gpu}'"
      f" --resource_gpu_count '{resource_gpu_count}'"
      f" --resource_timeout {resource_timeout}"
      f" --resource_max_retries {resource_max_retries}"
    )
    if trigger:
      reconstructed_cli_comm += " --trigger"
    if untracked:
      reconstructed_cli_comm += " --untracked"
    if untracked_no_limit:
      reconstructed_cli_comm += " --untracked_no_limit"
    if save_to_relic:
      reconstructed_cli_comm += " --save_to_relic"
    if enable_system_monitoring:
      reconstructed_cli_comm += " --enable_system_monitoring"
    if resource_max_retries < 1:
      logger.error(f"max_retries must be >= 1. Got: {resource_max_retries}\n  Fix: set --max_retries=2")
      raise ValueError()

    # uncommenting these since it can contain sensitive information, kept just in case for taking dictionaries as input
    # if relics_kwargs:
    #   rks = str(relics_kwargs).replace("'", "")
    #   reconstructed_cli_comm += f" --relics_kwargs '{rks}'"

    for k, v in run_kwargs.items():
      if type(v) == bool:
        reconstructed_cli_comm += f" --{k}"
      else:
        reconstructed_cli_comm += f" --{k} '{v}'"
    logger.debug(f"command: {reconstructed_cli_comm}")

    if untracked_no_limit and not untracked:
      logger.debug("untracked_no_limit is True but untracked is False. Setting untracked to True")
      untracked = True

    # first step is to get all the relevant information from the DB and servers
    workspace_id = workspace_id or secret(AuthConfig.workspace_id)
    project_name, project_id = get_project_name_id(project_name_or_id, workspace_id)
    job_name = "nbxj_" + project_name[:15]
    try:
      job = Job(job_name = job_name)
    except:
      logger.warn(f"Job not found. Creating a new job: {job_name}")
      job = Job.new(job_name = job_name, description = "automatically created by nbx LMAO for project: " + project_name)

    logger.info(f"Project: {project_name} ({project_id})")
    lmao_stub = get_lmao_stub()

    # create a call init run and get the experiment metadata
    init_folder, _ = os.path.split(init_path)
    init_folder = init_folder or "."
    if os.path.exists(U.join(init_folder, ".git")):
      git_det = get_git_details(init_folder)
    else:
      git_det = {}

    run = lmao_stub.init_run(InitRunRequest(
      workspace_id = workspace_id,
      project_name = project_name,
      project_id = project_id,
      agent_details = AgentDetails(
        type = AgentDetails.NBX.JOB,
        nbx_job_id = job.id, # run id will have to be updated from the job
      ),
      config = ExperimentConfig(
        run_kwargs = run_kwargs,
        git = git_det,
        resource = Resource(
          cpu = resource_cpu,
          memory = resource_memory,
          disk_size = resource_disk_size,
          gpu = resource_gpu,
          gpu_count = resource_gpu_count,
          timeout = resource_timeout,
          max_retries = resource_max_retries,
        ),
        cli_comm = reconstructed_cli_comm,
        save_to_relic = save_to_relic,
        enable_system_monitoring = enable_system_monitoring,
      ).to_json(),
    ))
    if not project_id:
      logger.info(f"Project {project_name} created with id: {run.project_id}")
    logger.info(f"Run ID: {run.experiment_id}")

    # connect to the relic
    r_keys = set(relics_kwargs.keys())
    valid_keys = {"bucket_name", "region", "nbx_resource_id", "nbx_integration_token"}
    extra_keys = r_keys - valid_keys
    if extra_keys:
      logger.error("Unknown arguments found:\n  * " + "\n  * ".join(extra_keys))
      raise RuntimeError("Unknown arguments found in the Relic")
    relic = Relics(LMAO_RELIC_NAME, workspace_id = workspace_id, create = True, **relics_kwargs)

    # create a git patch and upload it to relics
    if git_det:
      _zf = U.join(init_folder, "untracked.zip")
      zip_file = zipfile.ZipFile(_zf, "w", zipfile.ZIP_DEFLATED)
      patch_file = U.join(init_folder, "nbx_auto_patch.diff")
      f = open(patch_file, "w")
      Popen(shlex.split(f"git diff {' '.join(git_det['uncommited_files'])}"), stdout=f, stderr=sys.stderr).wait()
      f.close()
      zip_file.write(patch_file, arcname = patch_file)
      if untracked:
        untracked_files = git_det["untracked_files"] # what to do with these?
        if untracked_files:
          # see if any file is larger than 10MB and if so, warn the user
          warn_once = False
          for f in untracked_files:
            if os.path.getsize(f) > 1e7 and not untracked_no_limit:
              logger.warning(f"File: {f} is larger than 10MB and will not be available in sync")
              logger.warning("  Fix: use git to track small files, avoid large files")
              logger.warning("  Fix: nbox.Relics can be used to store large files")
              logger.warning("  Fix: use --untracked-no-limit to upload all files")
              warn_once = True
              continue
            zip_file.write(f, arcname = f)
      relic.put_to(_zf, f"{project_name}/{run.experiment_id}/git.zip")
      os.remove(_zf)
      os.remove(patch_file)

    # tell the server that this run is being scheduled so atleast the information is visible on the dashboard
    upload_job_folder(
      "job",
      init_folder = init_path,
      id = job.id,

      # pass along the resource requirements
      resource_cpu = resource_cpu,
      resource_memory = resource_memory,
      resource_disk_size = resource_disk_size,
      resource_gpu = resource_gpu,
      resource_gpu_count = resource_gpu_count,
      resource_timeout = resource_timeout,
      resource_max_retries = resource_max_retries,
    )

    if trigger:
      fp = f"{run.project_id}/{run.experiment_id}"
      tag = f"{LMAO_RM_PREFIX}{fp}"
      logger.debug(f"Running job '{job.name}' ({job.id}) with tag: {tag}")
      job.trigger(tag)

    # finally print the location of the run where the users can track this
    logger.info(f"Run location: {secret(AuthConfig.url)}/workspace/{run.workspace_id}/monitoring/{run.project_id}/{run.experiment_id}")

# do not change these it can become a huge pain later on
LMAO_RELIC_NAME = "experiments"
LMAO_RM_PREFIX = "NBXLmao-"
LMAO_ENV_VAR_PREFIX = "NBX_LMAO_"
