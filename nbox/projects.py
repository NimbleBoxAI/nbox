import os
import sys
import ast
import json
import shlex
import zipfile
from pprint import pformat
from subprocess import Popen

from google.protobuf import field_mask_pb2

from nbox.utils import logger
from nbox import utils as U
from nbox.auth import secret, AuthConfig, auth_info_pb
from nbox.init import nbox_ws_v1, nbox_model_service_stub
from nbox import messages as mpb
from nbox.lmao import (
  Lmao,
  LmaoLive,
  get_lmao_stub,
  get_git_details,
  lmao_v2_pb2 as pb,
  ExperimentConfig,
  LMAO_RM_PREFIX,
  LiveConfig,
  LMAO_SERVING_FILE,
)
from nbox.relics import Relics
from nbox.lmao.lmao_rpc_client import LMAO_Stub
from nbox.jobs import Job, Serve
from nbox.hyperloop.deploy import serve_pb2
from nbox.hyperloop.common.common_pb2 import Resource
from nbox.nbxlib.astea import Astea


def _parse_job_code(init_path: str, run_kwargs) -> tuple[str, dict]:
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
    if not type(v) == ast.Constant:
      raise Exception(f"Default value for {_k} is not a primitive type [int, float, str, bool] but {type(v)}")
    _default_kwarg[_k] = v.value
  args_set = set([x.arg for x in args])
  args_na = args_set - set(_default_kwarg.keys())
  args_not_passed = set(args_na) - set(run_kwargs.keys())
  if len(args_not_passed):
    raise Exception(f"Following arguments are not passed: {args_not_passed} but required, this can cause errors during execution")
  extra_args = set(run_kwargs.keys()) - args_set
  if len(extra_args) and not fn.node.args.kwarg:
    raise Exception(f"Following arguments are passed but not consumed by {fn_name}: {extra_args}")
  final_dict = {}
  for k,v in _default_kwarg.items():
    final_dict[k] = type(v)(run_kwargs.get(k, v))
  return final_dict, init_folder

class ProjectState:
  project_id: str = ""
  experiment_id: str = ""
  serving_id: str = ""

  def data():
    return {k:getattr(ProjectState, k) for k in ProjectState.__dict__ if not k.startswith("__") and k != "data"}


class Project:
  def __init__(self, id: str = ""):
    id = id or ProjectState.project_id
    if not id:
      raise ValueError("Project ID is not set")
    logger.info(f"Connecting to Project: {id}")
    self.stub = nbox_ws_v1.projects.u(id)
    self.data = self.stub()
    self.pid = self.data["project_id"]
    self.lmao_stub = get_lmao_stub()
    self.project = self.lmao_stub.get_project(pb.Project(
      workspace_id = secret.workspace_id,
      project_id = self.pid
    ))
    if self.project is None:
      raise ValueError("Could not connect to Monitoring backend.")
    self.relic = Relics(id = self.project.relic_id)
    self.workspace_id = secret.workspace_id

  def __repr__(self) -> str:
    return f"Project({self.pid})"

  @property
  def metadata(self):
    """A NimbleBox project is a very large entity and its components are in multiple places.
    `metadata` is a dictionary that contains all the information about the project."""
    return {"details": self.data, "lmao": mpb.MessageToDict(self.project, including_default_value_fields=True)}

  def put_settings(self, project_name: str = "", project_description: str = ""):
    self.stub(
      "put",
      project_name = project_name or self.data["project_name"],
      project_description = project_description or self.data["project_description"]
    )
    self.data["project_name"] = project_name or self.data["project_name"]
    self.data["project_description"] = project_description or self.data["project_description"]

  def get_lmao_stub(self) -> LMAO_Stub:
    return self.lmao_stub

  def get_exp_tracker(
      self,
      experiment_id: str = "",
      **kwargs,
    ) -> Lmao:
    lmao = Lmao(
      project_id = self.pid,
      experiment_id = experiment_id or ProjectState.experiment_id,
      **kwargs,
    )
    return lmao

  def get_live_tracker(self, serving_id: str = "", metadata = {}) -> LmaoLive:
    lmao = LmaoLive(
      project_id = self.pid,
      serving_id = serving_id or ProjectState.serving_id,
      metadata = metadata,
    )
    return lmao

  def get_relic(self) -> Relics:
    return self.relic

  def get_job_id(self) -> str:
    return self.data["job_list"][0]

  def get_deployment_id(self) -> str:
    return self.data["deployment_list"][0]

  # Now here's the thing about the project, project is supposed to be an end to end thing from training to deployment
  # however it depends on the NBX-Jobs and Deploy to provide the actual computation. NBX-Jobs and Deploy are application
  # agnostic meaning that they do not know what they are running. So in order to inform them what is going to happen
  # we use Environment Variables to pass the information.
  # 
  # Here's the flow in run:
  # - recreate the CLI command that was used to trigger the job
  # - get the exsting job at the project's associated job id
  # - [JOB ONLY] parse the init_path and get the git details
  # - LMAO initialise a new experiment
  # - upload the job code
  # - [JOB ONLY] if the git details are present then package the git files
  # - trigger with `RunTag`
  # 
  # Here's the flow in deploy:
  # - recreate the CLI command that was used to trigger the serving
  # - get the exsting serving at the project's associated serving id
  # - LMAO initialise the serving
  # - upload the serving code
  # - deploy with Tag

  def run(
    self,
    init_path: str,

    # all the arguments for the git files
    untracked: bool = False,
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
    workspace_id = secret(AuthConfig.workspace_id)

    # fix data type conversion caused by the CLI
    resource_cpu = str(resource_cpu)
    resource_memory = str(resource_memory)
    resource_disk_size = str(resource_disk_size)
    resource_gpu = str(resource_gpu)
    resource_gpu_count = str(resource_gpu_count)
    resource_max_retries = int(resource_max_retries)

    # reconstruct the entire CLI command so we can show it in the UI
    reconstructed_cli_comm = (
      f"nbx projects --id '{self.pid}' - run '{init_path}'" +
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
    run = self.lmao_stub.init_run(pb.InitRunRequest(
      workspace_id = workspace_id,
      project_id = self.pid,
      agent_details = pb.AgentDetails(
        type = pb.AgentDetails.NBX.JOB,
        nbx_job_id = job.id, # run id will have to be updated from the job
      ),
      config = ExperimentConfig(
        run_kwargs = run_kwargs,
        git = git_det,
        resource = job.job_proto.resource,
        cli_comm = reconstructed_cli_comm,
        save_to_relic = True,
        enable_system_monitoring = False,
      ).to_json(),
    ))
    logger.info(f"Created new experiment ID: {run.experiment_id}")

    # create a git patch and upload it to relics, this is unique to LMAO Run
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

      # upload the patch file to relics
      self.relic.put_to(_zf, f"{run.experiment_id}/git.zip")
      os.remove(_zf)
      os.remove(patch_file)

    # tell the server that this run is being scheduled so atleast the information is visible on the dashboard
    Job.upload(
      init_folder = init_path,
      id = job.id,

      # pass along the resource requirements
      resource_cpu = resource_cpu,
      resource_memory = resource_memory,
      resource_disk_size = resource_disk_size,
      resource_gpu = resource_gpu,
      resource_gpu_count = resource_gpu_count,
      resource_max_retries = resource_max_retries,
    )

    # now trigger the experiment
    fp = f"{run.project_id}/{run.experiment_id}"
    tag = f"{LMAO_RM_PREFIX}{fp}"
    logger.info(f"Running job '{job.name}' ({job.id}) with tag: {tag}")
    job.trigger(tag)

    # finally print the location of the run where the users can track this
    logger.info(f"Run location: {secret(AuthConfig.url)}/workspace/{run.workspace_id}/projects/{run.project_id}#Experiments")

  def serve(
    self,
    init_path: str,

    # all the things for resources
    resource_cpu: str = "",
    resource_memory: str = "",
    resource_disk_size: str = "",
    resource_gpu: str = "",
    resource_gpu_count: str = "",
    resource_max_retries: int = 0,

    # all the things for serving
    serving_type: str = "fastapi_v2",
  ):
    workspace_id = secret(AuthConfig.workspace_id)

    # fix data type conversion caused by the CLI
    resource_cpu = str(resource_cpu)
    resource_memory = str(resource_memory)
    resource_disk_size = str(resource_disk_size)
    resource_gpu = str(resource_gpu)
    resource_gpu_count = str(resource_gpu_count)
    resource_max_retries = int(resource_max_retries)

    # reconstruct the entire CLI command so we can show it in the UI
    reconstructed_cli_comm = (
      f"nbx projects --id '{self.pid}' - run '{init_path}'" +
      (f" --resource_cpu '{resource_cpu}'" if resource_cpu else "") +
      (f" --resource_memory '{resource_memory}'" if resource_memory else "") +
      (f" --resource_disk_size '{resource_disk_size}'" if resource_disk_size else "") +
      (f" --resource_gpu '{resource_gpu}'" if resource_gpu else "") +
      (f" --resource_gpu_count '{resource_gpu_count}'" if resource_gpu_count else "") +
      (f" --resource_max_retries {resource_max_retries}" if resource_max_retries else "")
    )
    logger.debug(f"command: {reconstructed_cli_comm}")

    # get the serving
    serving_pb = serve_pb2.Serving(id = self.get_deployment_id())
    serving_pb.resource.MergeFrom(Resource(
      cpu = resource_cpu,
      memory = resource_memory,
      disk_size = resource_disk_size,
      gpu = resource_gpu,
      gpu_count = resource_gpu_count,
      max_retries = resource_max_retries,
    ))

    # now we create a live tracker that can then be reused by the serving
    serving = self.lmao_stub.init_serving(pb.InitRunRequest(
      workspace_id = workspace_id,
      project_id = self.pid,
      agent_details = pb.AgentDetails(
        type = pb.AgentDetails.NBX.JOB,
        nbx_serving_id = serving_pb.id
      ),
      config = LiveConfig(
        resource = serving_pb.resource,
        cli_comm = reconstructed_cli_comm,
        enable_system_monitoring = False,
      ).to_json(),
    ))
    logger.info(f"Created new live tracking ID: {serving.serving_id}")

    # now upload this model via the Serve functionality
    deployment_model: Serve = Serve.upload(
      init_folder = init_path,
      id = serving_pb.id,
      trigger = False,
      resource_cpu = resource_cpu,
      resource_memory = resource_memory,
      resource_disk_size = resource_disk_size,
      resource_gpu = resource_gpu,
      resource_gpu_count = resource_gpu_count,
      model_name = serving.serving_id.replace("-", "_"),
      serving_type = serving_type,
      _ret = True
    )

    # deploy with the tag
    tag = f"{LMAO_RM_PREFIX}{self.pid}/{serving.serving_id}"
    logger.info(f"Deploying serving '{deployment_model.serving_id}' ({deployment_model.model_id}) with tag: {tag}")
    nbox_model_service_stub.Deploy(
      serve_pb2.ModelRequest(
        model = serve_pb2.Model(
          id = deployment_model.serving_id,
          serving_group_id = deployment_model.serving_id,
          feature_gates = {
            "SetModelMetadata": tag
          }
        ),
        auth_info = auth_info_pb(),
      ),
    )
