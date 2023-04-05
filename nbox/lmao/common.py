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

import re
import json
from git import Repo
from typing import Union, Dict, Any

from nbox.utils import logger
from nbox.auth import secret, AuthConfig
from nbox.init import nbox_ws_v1
from nbox.lmao.lmao_rpc_client import (
  LMAO_Stub, # main stub class
  Record,
  ListProjectsRequest,
  Project as ProjectProto,
)
from nbox.hyperloop.common.common_pb2 import Resource
from nbox import messages as mpb

"""
functional components of LMAO
"""

def get_lmao_stub() -> LMAO_Stub:
  # lmao_stub = LMAO_Stub(url = "http://localhost:8080/monitoring", session = nbox_ws_v1._session)
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

def get_project(project_id: str) -> ProjectProto:
  lmao_stub = get_lmao_stub()
  project: ProjectProto = lmao_stub.get_project(ListProjectsRequest(
    workspace_id = secret(AuthConfig.workspace_id),
    project_id_or_name = project_id
  ))
  return project


def resource_from_dict(d: Dict[str, Any]):
  return Resource(
    cpu = str(d["cpu"]),
    memory = str(d["memory"]),
    disk_size = str(d["disk_size"]),
    gpu = str(d["gpu"]),
    gpu_count = str(d["gpu_count"]),
    timeout = int(d["timeout"]),
    max_retries = int(d["max_retries"]),
  )

"""
Common structs
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
      "resource": mpb.message_to_dict(self.resource),
      "cli_comm": self.cli_comm,
      "save_to_relic": self.save_to_relic,
      "enable_system_monitoring": self.enable_system_monitoring,
    }

  def to_json(self):
    return json.dumps(self.to_dict())

  @classmethod
  def from_json(cls, json_str):
    d = json.loads(json_str)
    d["resource"] = resource_from_dict(d["resource"])
    return cls(**d)



class LiveConfig:
  def __init__(
    self,
    resource: Resource,
    cli_comm: str,
    enable_system_monitoring: bool = False,
  ):
    self.resource = resource
    self.cli_comm = cli_comm
    self.enable_system_monitoring = enable_system_monitoring
    self.keys = set()

  def to_dict(self):
    out = {
      "resource": mpb.message_to_dict(self.resource),
      "cli_comm": self.cli_comm,
      "enable_system_monitoring": self.enable_system_monitoring,
    }
    for k in self.keys:
      out[k] = getattr(self, k)
    return out
  
  def to_json(self):
    return json.dumps(self.to_dict())
  
  @classmethod
  def from_json(cls, json_str) -> 'LiveConfig':
    d = json.loads(json_str)
    d["resource"] = resource_from_dict(d["resource"])
    _cls = cls(**d)
    for k in d:
      if k not in ["resource", "cli_comm", "enable_system_monitoring"]:
        _cls.add(k, d[k])
  
  def add(self, key, value):
    setattr(self, key, value)
    self.keys.add(key)

  def get(self, key):
    return getattr(self, key)


"""
Constants
"""

# do not change these it can become a huge pain later on
LMAO_RELIC_NAME = "experiments"
LMAO_RM_PREFIX = "NBXLmao-"
LMAO_SERVING_FILE = "NBXLmaoServingCfg.json"
LMAO_ENV_VAR_PREFIX = "NBX_LMAO_"
