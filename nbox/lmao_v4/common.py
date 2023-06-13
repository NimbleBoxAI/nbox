import re
import grpc
import json
import collections
from git import Repo
from functools import lru_cache
from typing import Union, Dict, Any

from nbox.auth import inside_pod, secret
from nbox.init import MetadataInjectInterceptor, WorkspaceIdInjectInterceptor
from nbox.utils import logger, lo
from nbox import messages as mpb
from nbox.hyperloop.common.common_pb2 import Resource

from nbox.lmao_v4.proto.lmao_service_pb2_grpc import LMAOStub
from nbox.lmao_v4.proto.logs_pb2 import TrackerLog
from nbox.lmao_v4.proto.project_pb2 import Project as ProjectProto

@lru_cache(1)
def get_lmao_stub() -> LMAOStub:
  token_cred = grpc.access_token_call_credentials(secret.access_token)
  ssl_creds = grpc.ssl_channel_credentials()
  creds = grpc.composite_channel_credentials(ssl_creds, token_cred)
  channel = grpc.secure_channel(secret.nbx_url.replace("https://", "dns:/") + ":443", creds)

  # channel = grpc.insecure_channel("dns:/0.0.0.0:50051") # debug mode

  future = grpc.channel_ready_future(channel)
  future.add_done_callback(lambda _: logger.debug(f"NBX 'LMAO' gRPC stub is ready!"))
  interceptors = [
    MetadataInjectInterceptor(),
    WorkspaceIdInjectInterceptor(),
  ]
  channel = grpc.intercept_channel(channel, *interceptors)
  stub = LMAOStub(channel)
  return stub

valid_key_regex = re.compile(r'[a-zA-Z0-9_\-\.]+$') 

def flatten(d, parent_key='', sep='.'):
  """flatten a dictionary with prefix and seperator, if """
  items = []
  for k, v in d.items():
    if sep in k:
      raise ValueError(f"Key '{k}' contains the seperator '{sep}', this can cause issue during rebuilding")
    new_key = parent_key + sep + k if parent_key else k
    if isinstance(v, collections.abc.MutableMapping):
      items.extend(flatten(v, new_key, sep=sep).items())
    else:
      items.append((new_key, v))
  return dict(items)


def get_tracker_log(log: dict) -> TrackerLog:
  items = flatten(log).items()
  out = TrackerLog()
  for k,v in items:
    if valid_key_regex.match(k) is None:
      raise ValueError(f"Invalid key '{k}', allowed letters digits and '_-'")
    if type(v) in (float, int):
      out.number_keys.append(k)
      out.number_values.append(v)
    elif type(v) == str:
      out.text_keys.append(k)
      out.text_values.append(v)
    else:
      raise ValueError(f"Invalid type '{type(v)}' for key '{k}'")
  return out


# there are some legacy functions that were built for lmao_v2, in some cases retrofitted for v4

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
  project: ProjectProto = lmao_stub.GetProject(ProjectProto(id=project_id))
  return project


def resource_from_dict(d: Dict[str, Any]):
  return Resource(
    cpu = str(d.get("cpu", "")),
    memory = str(d.get("memory", "")),
    disk_size = str(d.get("disk_size", "")),
    gpu = str(d.get("gpu", "")),
    gpu_count = str(d.get("gpu_count", "")),
    timeout = int(d.get("timeout", 0)),
    max_retries = int(d.get("max_retries", 0)),
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
  
  @classmethod
  def from_dict(cls, data):
    if not isinstance(data, Resource):
      data["resource"] = resource_from_dict(data["resource"])
    return cls(**data)

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
    extra_kwargs: Dict[str, Any] = {},
  ):
    self.resource = resource
    self.cli_comm = cli_comm
    self.enable_system_monitoring = enable_system_monitoring
    self.extra_kwargs = extra_kwargs

  def to_dict(self):
    return {
      "resource": mpb.message_to_dict(self.resource),
      "cli_comm": self.cli_comm,
      "enable_system_monitoring": self.enable_system_monitoring,
      "extra_kwargs": self.extra_kwargs,
    }
  
  def to_json(self):
    return json.dumps(self.to_dict())
  
  @classmethod
  def from_json(cls, json_str) -> 'LiveConfig':
    d = json.loads(json_str)
    d["resource"] = resource_from_dict(d["resource"])
    _cls = cls(**d)
    return _cls


"""
Constants
"""

# do not change these it can become a huge pain later on
LMAO_RELIC_NAME = "experiments"
LMAO_RM_PREFIX = "NBXLmao-"
LMAO_SERVING_FILE = "NBXLmaoServingCfg.json"
LMAO_ENV_VAR_PREFIX = "NBX_LMAO_"
