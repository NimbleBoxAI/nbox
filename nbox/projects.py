import os
import sys
import ast
import shlex
import zipfile
from subprocess import Popen
from typing import Tuple, Dict

from nbox import utils as U
from nbox.auth import secret
from nbox.utils import logger, lo
from nbox.relics import Relics
from nbox import messages as mpb
from nbox.init import nbox_ws_v1
from nbox.jd_core import Job, Serve, upload_job_folder

from nbox.nbxlib.astea import Astea

from nbox.hyperloop.common.common_pb2 import Resource
from nbox.lmao_v4 import get_lmao_stub, get_git_details, LMAO_RM_PREFIX, ExperimentConfig, Tracker
from nbox.lmao_v4.proto.lmao_service_pb2_grpc import LMAOStub
from nbox.lmao_v4.proto import tracker_pb2 as t_pb
from nbox.lmao_v4.proto import project_pb2 as p_pb


def evaluate_unary_op(node):
  if isinstance(node.op, ast.USub):  # Unary negation
    return -evaluate_node(node.operand)
  elif isinstance(node.op, ast.UAdd):  # Unary positive
    return evaluate_node(node.operand)
  elif isinstance(node.op, ast.Not):  # Logical not
    return not evaluate_node(node.operand)
  else:
    raise ValueError("Unsupported unary operator")

def evaluate_node(node):
  if isinstance(node, ast.Constant):
    return node.value
  elif isinstance(node, ast.UnaryOp):
    return evaluate_unary_op(node)
  else:
    raise ValueError("Unsupported node type")

def _parse_job_code(init_path: str, run_kwargs) -> Tuple[str, Dict]:
  # analyse the input file and function, extract the types and build the run_kwargs
  fn_file, fn_name = init_path.split(":")
  if not os.path.exists(fn_file + ".py"):
    raise Exception(f"File {fn_file} does not exist")
  init_folder, fn_file = os.path.split(fn_file + ".py")
  init_folder = init_folder or "."
  tea = Astea(fn_file)
  items = tea.find(fn_name)
  if not len(items):
    raise Exception(f"Function {fn_name} not found in {fn_file}")
  fn = items[0]
  defaults = fn.node.args.defaults
  args = fn.node.args.args
  _default_kwarg = {}
  for k,v in zip(args[::-1], defaults[::-1]):
    _k = k.arg
    if type(v) == ast.UnaryOp:
      logger.warning(f"Default value for var '{_k}' is a UnaryOp, cannot validate the type")
      try:
        evaluate_node(v)
      except:
        raise Exception(f"Default value for var '{_k}' is not a primitive type [int, float, str, bool, None] but {type(v)}")
      _default_kwarg[_k] = evaluate_node(v)
    elif not type(v) == ast.Constant:
      raise Exception(f"Default value for var '{_k}' is not a primitive type [int, float, str, bool, None] but {type(v)}")
    else:
      try:
        _default_kwarg[_k] = v.value
      except:
        raise Exception(f"Default value for var '{_k}' is not a primitive type [int, float, str, bool, None] but {type(v)}")
  args_set = set([x.arg for x in args])
  args_na = args_set - set(_default_kwarg.keys())
  args_not_passed = set(args_na) - set(run_kwargs.keys())
  if len(args_not_passed):
    raise Exception(f"Following arguments are not passed: {args_not_passed} but required, this can cause errors during execution")
  extra_args = set(run_kwargs.keys()) - args_set
  if len(extra_args) and not fn.node.args.kwarg:
    raise Exception(f"Following arguments are passed but not consumed by {fn_name}: {extra_args}")
  final_dict = {}

  # left merge the two dicts: default_kwargs <- run_kwargs
  merged = {**_default_kwarg, **run_kwargs}
  for k,v in merged.items():
    if type(v) == type(None):
      final_dict[k] = run_kwargs.get(k, v)
    else:
      final_dict[k] = type(v)(run_kwargs.get(k, v))
  return final_dict, init_folder



### ---------------
# this will be eventually merged with the project in the root scope


_SUPPORTED_SERVER_TYPES = ["fastapi"]

class ProjectState:
  project_id: str = ""
  experiment_id: str = ""
  serving_id: str = ""

  def data():
    return {k:getattr(ProjectState, k) for k in ProjectState.__dict__ if not k.startswith("__") and k != "data"}


# aka Project_v4
# TODO: @yashbonde: all the jobs are Job gRPC stubs
# TODO: @yashbonde: all the experiments are Experiment gRPC stubs
class Project:
  def __init__(self, id: str = ""):
    id = id or ProjectState.project_id
    if not id:
      raise ValueError("Project ID is not set")
    logger.info(f"Connecting to Project: {id}")

    self.stub = nbox_ws_v1.projects.u(id)
    self.data = self.stub()
    self.lmao_stub: LMAOStub = get_lmao_stub()
    _p = p_pb.Project(
      id = id
    )
    self.project_pb = self.lmao_stub.GetProject(_p)
    if self.project_pb is None:
      raise ValueError("Could not connect to Monitoring backend.")
    self.artifact = Relics(id = self.project_pb.relic_id)
    self.workspace_id = secret.workspace_id

  def __repr__(self) -> str:
    return f"Project({self.project_pb.id})"

  # things about the project as a resource

  def list_trackers(self, live: bool = False, status: str = t_pb.Tracker.Status.UNSET_STATUS) -> t_pb.ListTrackersResponse:
    req = t_pb.ListTrackersRequest(
      project_id = self.project_pb.id,
      status = status,
      tracker_type = t_pb.TrackerType.LIVE if live else t_pb.TrackerType.EXPERIMENT,
    )
    return self.lmao_stub.ListTrackers(req)

  @property
  def metadata(self):
    """A NimbleBox project is a very large entity and its components are in multiple places.
    `metadata` is a dictionary that contains all the information about the project."""
    return {"details": self.data, "lmao": mpb.MessageToDict(self.project_pb, including_default_value_fields=True)}

  def put_settings(self, project_name: str = "", project_description: str = ""):
    self.stub(
      "put",
      project_name = project_name or self.data["project_name"],
      project_description = project_description or self.data["project_description"]
    )
    self.data["project_name"] = project_name or self.data["project_name"]
    self.data["project_description"] = project_description or self.data["project_description"]

  # some getters for underlying objects

  def get_lmao_stub(self) -> LMAOStub:
    return self.lmao_stub
  
  def get_artifact(self) -> Relics:
    return self.artifact

  def get_job_id(self) -> str:
    return self.data["job_list"][0]

  def get_deployment_id(self) -> str:
    return self.data["deployment_list"][0]

  def get_tracker(self, tracker_type: int, tracker_id: str, config) -> Tracker:
    return Tracker(
      project_id = self.project_pb,
      tracker_id = tracker_id,
      config = config,
      live_tracker = tracker_type == t_pb.TrackerType.LIVE,
    )

  def get_exp_tracker(
      self,
      experiment_id: str = "",
      metadata = {},
      update_agent: bool = False,
    ) -> Tracker:
    if not experiment_id and ProjectState.experiment_id:
      experiment_id = ProjectState.experiment_id
    tracker = self.get_tracker(
      tracker_type = t_pb.TrackerType.EXPERIMENT,
      tracker_id = experiment_id,
      config = metadata
    )
    if update_agent:
      tracker.update_tracker_agent()
    return tracker

  def get_live_tracker(self, serving_id: str = "", metadata = {}) -> Tracker:
    return self.get_tracker(
      tracker_type = t_pb.TrackerType.LIVE,
      tracker_id = serving_id,
      config = metadata
    )

  # Some big things primarily built for the cli but can be reused as APIs as well

  def run(
    self,
    init_path: str,

    # all the arguments for the git files
    untracked: bool = False,
    upload_git: bool = False,
    untracked_no_limit: bool = False,

    # all the things for resources
    resource_cpu: str = "",
    resource_memory: str = "",
    resource_disk_size: str = "",
    resource_gpu: str = "",
    resource_gpu_count: str = "",
    resource_max_retries: int = 0,

    # any other things to pass to the function / class being called
    **run_kwargs,
  ):
    """Upload and register a new run for a NBX-LMAO project.

    Args:
      init_path (str): This can be a path to a `folder` or can be optionally of the structure `fp:fn` where `fp` is the path to the file and `fn` is the function name.
      untracked (bool, optional): If True, then untracked files below 1MB will also be zipped and uploaded. Defaults to False.
      untracked_no_limit (bool, optional): If True, then all untracked files will also be zipped and uploaded. Defaults to False.
      resource_cpu (str, optional): The CPU resource to allocate to the run. It can be found on your NimbleBox dashboard.
      resource_memory (str, optional): The memory resource to allocate to the run. It can be found on your NimbleBox dashboard.
      resource_disk_size (str, optional): The disk size resource to allocate to the run. It can be found on your NimbleBox dashboard.
      resource_gpu (str, optional): The GPU resource to use. It can be found on your NimbleBox dashboard.
      resource_gpu_count (str, optional): Number of GPUs allocated to the experiment. It can be found on your NimbleBox dashboard.
      resource_max_retries (int, optional): The maximum number of retries for a run. It can be found on your NimbleBox dashboard.
      **run_kwargs: These are the kwargs that will be passed to your function.
    """
    # fix data type conversion caused by the CLI
    resource_cpu = str(resource_cpu)
    resource_memory = str(resource_memory)
    resource_disk_size = str(resource_disk_size)
    resource_gpu = str(resource_gpu)
    resource_gpu_count = str(resource_gpu_count)
    resource_max_retries = int(resource_max_retries)

    # reconstruct the entire CLI command so we can show it in the UI
    reconstructed_cli_comm = (
      f"nbx projects --id '{self.project_pb.id}' - run '{init_path}'" +
      (f" --resource_cpu '{resource_cpu}'" if resource_cpu else "") +
      (f" --resource_memory '{resource_memory}'" if resource_memory else "") +
      (f" --resource_disk_size '{resource_disk_size}'" if resource_disk_size else "") +
      (f" --resource_gpu '{resource_gpu}'" if resource_gpu else "") +
      (f" --resource_gpu_count '{resource_gpu_count}'" if resource_gpu_count else "") +
      (f" --resource_max_retries {resource_max_retries}" if resource_max_retries else "")
    )
    if untracked_no_limit and not untracked:
      raise ValueError("Cannot set untracked_no_limit to True if untracked is False")
    if untracked:
      reconstructed_cli_comm += " --untracked"
    if untracked_no_limit:
      reconstructed_cli_comm += " --untracked_no_limit"
    for k, v in run_kwargs.items():
      if type(v) == bool:
        reconstructed_cli_comm += f" --{k}"
      elif type(v) == str:
        reconstructed_cli_comm += f" --{k} '{v}'"
      elif isinstance(v, (int, float)):
        reconstructed_cli_comm += f" --{k} {v}"
      elif isinstance(v, (list, tuple)):
        reconstructed_cli_comm += f" --{k} \"[{v}]\""
      else:
        reconstructed_cli_comm += f" --{k} '{v}'"
    logger.debug(f"command: {reconstructed_cli_comm}")

    # get the job
    job = Job(job_id = self.get_job_id())
    job.job_proto.resource.MergeFrom(Resource(
      cpu = resource_cpu,
      memory = resource_memory,
      disk_size = resource_disk_size,
      gpu = resource_gpu,
      gpu_count = resource_gpu_count,
      max_retries = resource_max_retries,
    ))

    # parse the code file + get git details, these two steps are unique to LMAO Run
    run_kwargs, init_folder = _parse_job_code(init_path = init_path, run_kwargs = run_kwargs) 
    if os.path.exists(U.join(init_folder, ".git")):
      git_det = get_git_details(init_folder)
    else:
      git_det = {}

    # initialise the run with the big object and get the experiment_id
    exp_config = ExperimentConfig(
      run_kwargs = run_kwargs,
      git = git_det,
      resource = job.job_proto.resource,
      cli_comm = reconstructed_cli_comm,
      save_to_relic = True,
      enable_system_monitoring = False,
    ).to_dict()
    logger.debug(lo("Experiment Config", **exp_config))
    exp_tracker = self.get_exp_tracker(metadata = exp_config)
    tracker_pb = exp_tracker.tracker_pb
    logger.info(f"Created new tracker ID: {tracker_pb.id}")

    # create a git patch and upload it to relics, this is unique to LMAO Run
    if git_det and upload_git:
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
          for f in untracked_files:
            if os.path.getsize(f) > 1e7 and not untracked_no_limit:
              logger.warning(f"File: {f} is larger than 10MB and will not be available in sync")
              logger.warning("  Fix: use git to track small files, avoid large files")
              logger.warning("  Fix: nbox.Relics can be used to store large files")
              logger.warning("  Fix: use --untracked-no-limit to upload all files")
              continue
            zip_file.write(f, arcname = f)

      # upload the patch file to relics
      self.artifact.put_to(_zf, f"{tracker_pb.id}/git.zip")
      os.remove(_zf)
      os.remove(patch_file)

    # tell the server that this run is being scheduled so atleast the information is visible on the dashboard
    upload_job_folder(
      method = "job",
      init_folder = init_path,
      project_id = self.project_pb.id,

      # pass along the resource requirements
      resource_cpu = resource_cpu,
      resource_memory = resource_memory,
      resource_disk_size = resource_disk_size,
      resource_gpu = resource_gpu,
      resource_gpu_count = resource_gpu_count,
      resource_max_retries = resource_max_retries,
    )

    # now trigger the experiment
    fp = f"{tracker_pb.project_id}/{tracker_pb.id}"
    tag = f"{LMAO_RM_PREFIX}{fp}"
    logger.info(f"Running job '{job.name}' ({job.id}) with tag: {tag}")
    job.trigger(tag)

    # finally print the location of the run where the users can track this
    logger.info(f"Run location: {secret.nbx_url}/workspace/{secret.workspace_id}/projects/{self.project_pb.id}#Experiments")

  def serve(
    self,
    init_path: str,
    server_type: str = "fastapi",

    # all the things for resources
    resource_cpu: str = "",
    resource_memory: str = "",
    resource_disk_size: str = "",
    resource_gpu: str = "",
    resource_gpu_count: str = "",
    resource_max_retries: int = 0,
  ):
    raise NotImplementedError("This is not implemented yet")
    if server_type not in _SUPPORTED_SERVER_TYPES:
      raise ValueError(f"server_type must be one of {_SUPPORTED_SERVER_TYPES}")
