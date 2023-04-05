"""
`nbox.Job` and `nbox.Serve` are wrappers to the NBX-Jobs and NBX-Deploy APIs and contains staticmethods for convinience from the CLI.

* `datetime.now(timezone.utc)` is incorrect, use [this](https://blog.ganssle.io/articles/2019/11/utcnow.html) method.
"""

import os
import sys
import tabulate
from typing import Tuple, List, Dict
from functools import lru_cache, partial
from datetime import datetime, timedelta, timezone
from google.protobuf.timestamp_pb2 import Timestamp
from google.protobuf.field_mask_pb2 import FieldMask

import nbox.utils as U
from nbox.auth import secret, AuthConfig, auth_info_pb
from nbox.utils import logger
from nbox.version import __version__
from nbox.messages import rpc, streaming_rpc
from nbox.init import nbox_grpc_stub, nbox_ws_v1, nbox_serving_service_stub, nbox_model_service_stub
from nbox.nbxlib.astea import Astea, IndexTypes as IT

from nbox.hyperloop.jobs.nbox_ws_pb2 import JobRequest
from nbox.hyperloop.jobs.job_pb2 import Job as JobProto
from nbox.hyperloop.jobs.dag_pb2 import DAG as DAGProto
from nbox.hyperloop.common.common_pb2 import Resource
from nbox.hyperloop.jobs.nbox_ws_pb2 import ListJobsRequest, ListJobsResponse, UpdateJobRequest
from nbox.hyperloop.deploy.serve_pb2 import ServingListResponse, ServingRequest, Serving, ServingListRequest, ModelRequest, Model as ModelProto, UpdateModelRequest


DEFAULT_RESOURCE = Resource(
  cpu = "128m",         # 100mCPU
  memory = "256Mi",     # MiB
  disk_size = "3Gi",    # GiB
  gpu = "none",         # keep "none" for no GPU
  gpu_count = "0",      # keep "0" when no GPU
  timeout = 120_000,    # 2 minutes between attempts
  max_retries = 2,      # third times the charm :P
)


class Schedule:
  def __init__(
    self,
    hour: int  = None,
    minute: int = None,
    days: list = [],
    months: list = [],
    starts: datetime = None,
    ends: datetime = None,
  ):
    """Make scheduling natural. Uses 24-hour nomenclature.

    Args:
      hour (int): Hour of the day, if only this value is passed it will run every `hour`
      minute (int): Minute of the hour, if only this value is passed it will run every `minute`
      days (str/list, optional): List of days (first three chars) of the week, if not passed it will run every day.
      months (str/list, optional): List of months (first three chars) of the year, if not passed it will run every month.
      starts (datetime, optional): UTC Start time of the schedule, if not passed it will start now.
      ends (datetime, optional): UTC End time of the schedule, if not passed it will end in 7 days.

    Examples:

      # 4:20PM everyday
      Schedule(16, 20)

      # 4:20AM every friday
      Schedule(4, 20, ["fri"])

      # 4:20AM every friday from jan to feb
      Schedule(4, 20, ["fri"], ["jan", "feb"])

      # 4:20PM everyday starting in 2 days and runs for 3 days
      starts = datetime.now(timezone.utc) + timedelta(days = 2) # NOTE: that time is in UTC
      Schedule(16, 20, starts = starts, ends = starts + timedelta(days = 3))

      # Every 1 hour
      Schedule(1)

      # Every 69 minutes
      Schedule(minute = 69)
    """
    self.hour = hour
    self.minute = minute

    self._is_repeating = self.hour or self.minute
    self.mode = None
    if self.hour == None and self.minute == None:
      raise ValueError("Atleast one of hour or minute should be passed")
    elif self.hour != None and self.minute != None:
      assert self.hour in list(range(0, 24)), f"Hour must be in range 0-23, got {self.hour}"
      assert self.minute in list(range(0, 60)), f"Minute must be in range 0-59, got {self.minute}"
    elif self.hour != None:
      assert self.hour in list(range(0, 24)), f"Hour must be in range 0-23, got {self.hour}"
      self.mode = "every"
      self.minute = datetime.now(timezone.utc).strftime("%m") # run every this minute past this hour
      self.hour = f"*/{self.hour}"
    elif self.minute != None:
      self.hour = self.minute // 60
      assert self.hour in list(range(0, 24)), f"Hour must be in range 0-23, got {self.hour}"
      self.minute = f"*/{self.minute % 60}"
      self.hour = f"*/{self.hour}" if self.hour > 0 else "*"
      self.mode = "every"

    _days = {k:str(i) for i,k in enumerate(["sun","mon","tue","wed","thu","fri","sat"])}
    _months = {k:str(i+1) for i,k in enumerate(["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"])}

    if isinstance(days, str):
      days = [days]
    if isinstance(months, str):
      months = [months]

    diff = set(days) - set(_days.keys())
    if len(diff):
      raise ValueError(f"Invalid days: {diff}")
    self.days = ",".join([_days[d] for d in days]) if days else "*"

    diff = set(months) - set(_months.keys())
    if len(diff):
      raise ValueError(f"Invalid months: {diff}")
    self.months = ",".join([_months[m] for m in months]) if months else "*"

    self.starts = starts or datetime.now(timezone.utc)
    self.ends = ends or datetime.now(timezone.utc) + timedelta(days = 7)

  @property
  def cron(self):
    """Get the cron string for the given schedule"""
    if self.mode == "every":
      return f"{self.minute} {self.hour} * * *"
    return f"{self.minute} {self.hour} * {self.months} {self.days}"

  def get_dict(self):
    """Get the dictionary representation of this Schedule"""
    return {"cron": self.cron, "mode": self.mode, "starts": self.starts, "ends": self.ends}

  def get_message(self) -> JobProto.Schedule:
    """Get the JobProto.Schedule object for this Schedule"""
    _starts = Timestamp()
    _starts.GetCurrentTime()
    _ends = Timestamp()
    _ends.FromDatetime(self.ends)
    return JobProto.Schedule(
      start = _starts,
      end = _ends,
      cron = self.cron
    )

  def __repr__(self):
    return str(self.get_dict())



################################################################################
"""
# Common Functions

These functions are common to both NBX-Jobs and NBX-Deploy.
"""
################################################################################

def upload_job_folder(
  method: str,
  init_folder: str,
  id: str = "",
  project_id: str = "",
  trigger: bool = False,

  # all the things for resources
  resource_cpu: str = "",
  resource_memory: str = "",
  resource_disk_size: str = "",
  resource_gpu: str = "",
  resource_gpu_count: str = "",
  resource_timeout: int = 0,
  resource_max_retries: int = 0,

  # deployment specific
  model_name: str = "",

  # X-type
  serving_type: str = "nbox",

  # there's no more need to pass the workspace_id anymore
  workspace_id: str = "",

  # some extra things for functionality
  _ret: bool = False,

  # finally everything else is assumed to be passed to the initialisation script
  **init_kwargs
):
  """Upload the code for a job or serving to the NBX.

  ### Engineer's Note

  This function is supposed to be exposed via CLI, you can of course make a programtic call to this as well. This is a reason
  why this function can keep on taking many arguments. However with greatm number of arguments comes great responsibility, ie.
  lots of if/else conditions. Broadly speaking this should manage all the complexity and pass along simple reduced intstructions
  to the underlying methods. Currently the arguments that are required in Jinja templates are packed as `exe_jinja_kwargs` and
  we call the `deploy_job` and `deploy_serving`.

  Args:
    method (str): The method to use, either "job" or "serving"
    init_folder (str): folder with all the relevant files or ``file_path:fn_name`` pair so you can use it as the entrypoint.
    name (str, optional): Name of the job. Defaults to "".
    id (str, optional): ID of the job. Defaults to "".
    project_id (str, optional): Project ID, if None uses the one from config. Defaults to "".
    trigger (bool, optional): If uploading a "job" trigger the job after uploading. Defaults to False.
    resource_cpu (str, optional): CPU resource. Defaults to "100m".
    resource_memory (str, optional): Memory resource. Defaults to "128Mi".
    resource_disk_size (str, optional): Disk size resource. Defaults to "3Gi".
    resource_gpu (str, optional): GPU resource. Defaults to "none".
    resource_gpu_count (str, optional): GPU count resource. Defaults to "0".
    resource_timeout (int, optional): Timeout resource. Defaults to 120_000.
    resource_max_retries (int, optional): Max retries resource. Defaults to 2.
    cron (str, optional): Cron string for scheduling. Defaults to "".
    workspace_id (str, optional): Workspace ID, if None uses the one from config. Defaults to "".
    init_kwargs (dict): kwargs to pass to the `init` function / class, if possible
  """
  from nbox.network import deploy_job, deploy_serving
  import nbox.nbxlib.operator_spec as ospec
  from nbox.nbxlib.serving import SupportedServingTypes as SST
  from nbox.projects import Project
  
  OT = ospec.OperatorType

  if method not in OT._valid_deployment_types():
    raise ValueError(f"Invalid method: {method}, should be either {OT._valid_deployment_types()}")
  # if (not name and not id) or (name and id):
  #   raise ValueError("Either --name or --id must be present")
  if trigger and method not in [OT.JOB, OT.SERVING]:
    raise ValueError(f"Trigger can only be used with '{OT.JOB}' or '{OT.SERVING}'")
  if model_name and method != OT.SERVING:
    raise ValueError(f"model_name can only be used with '{OT.SERVING}'")
  
  # get the correct ID based on the project_id
  if (not project_id and not id) or (project_id and id):
    raise ValueError("Either --project-id or --id must be present")
  if project_id:
    p = Project(project_id)
    if method == OT.JOB:
      id = p.get_job_id()
    else:
      id = p.get_deployment_id()
    logger.info(f"Using project_id: {project_id}, found id: {id}")

  if ":" not in init_folder:
    # this means we are uploading a traditonal folder that contains a `nbx_user.py` file
    # in this case the module is loaded on the local machine and so user will need to have
    # everything installed locally. This was a legacy method before 0.10.0
    logger.error(
      'Old method of having a manual nbx_user.py file is now deprecated\n'
      f'  Fix: nbx {method} upload file_path:fn_cls_name --id "id"'
    )
    raise ValueError("Old style upload is not supported anymore")

  # In order to upload we can either chose to upload the entire folder, but can we implement a style where
  # only that specific function is uploaded? This is useful for raw distributed compute style.
  commands = init_folder.split(":")
  if len(commands) == 2:
    fn_file, fn_name = commands
    mode = "folder"
  elif len(commands) == 3:
    mode, fn_file, fn_name = commands
    if mode not in ["file", "folder"]:
      raise ValueError(f"Invalid mode: '{mode}' in upload command, should be either 'file' or 'folder'")
  else:
    raise ValueError(f"Invalid init_folder: {init_folder}")
  if mode != "folder":
    raise NotImplementedError(f"Only folder mode is supported, got: {mode}")
  if not os.path.exists(fn_file+".py"):
    raise ValueError(f"File {fn_file}.py does not exist")
  init_folder, file_name = os.path.split(fn_file)
  init_folder = init_folder or "."
  fn_name = fn_name.strip()
  if not os.path.exists(init_folder):
    raise ValueError(f"Folder {init_folder} does not exist")
  logger.info(f"Uploading code from folder: {init_folder}:{file_name}:{fn_name}")
  _curdir = os.getcwd()
  os.chdir(init_folder)

  workspace_id = workspace_id or secret(AuthConfig.workspace_id)

  # Now that we have the init_folder and function name, we can throw relevant errors
  perform_tea = True
  if method == OT.SERVING:
    if serving_type not in SST.all():
      raise ValueError(f"Invalid serving_type: {serving_type}, should be one of {SST.all()}")
    if serving_type == SST.FASTAPI or serving_type == SST.FASTAPI_V2:
      logger.warning(f"You have selected serving_type='{serving_type}', this assumes the object: {fn_name} is a FastAPI app")
      init_code = fn_name
      perform_tea = False
      load_operator = False

  if perform_tea:
    # build an Astea and analyse it for getting the computation that is going to be run
    load_operator = True
    tea = Astea(fn_file+".py")
    items = tea.find(fn_name, [IT.CLASS, IT.FUNCTION])
    if len(items) > 1:
      raise ModuleNotFoundError(f"Multiple {fn_name} found in {fn_file}.py")
    elif len(items) == 0:
      logger.error(f"Could not find function or class type: '{fn_name}'")
      raise ModuleNotFoundError(f"Could not find function or class type: '{fn_name}'")
    fn = items[0]
    if fn.type == IT.FUNCTION:
      # does not require initialisation
      if len(init_kwargs):
        logger.error(
          f"Cannot pass kwargs to a function: '{fn_name}'\n"
          f"  Fix: you cannot pass kwargs {set(init_kwargs.keys())} to a function"
        )
        raise ValueError("Function does not require initialisation")
      init_code = fn_name
    elif fn.type == IT.CLASS:
      # requires initialisation, in this case we will store the relevant to things in a Relic
      init_comm = ",".join([f"{k}={v}" for k, v in init_kwargs.items()])
      init_code = f"{fn_name}({init_comm})"
      logger.info(f"Starting with init code:\n  {init_code}")

  # load up the things that are to be passed to the exe.py file
  exe_jinja_kwargs = {
    "file_name": file_name,
    "fn_name": fn_name,
    "init_code": init_code,
    "load_operator": load_operator,
    "serving_type": serving_type,
  }

  # create a requirements.txt file if it doesn't exist with the latest nbox version
  if not os.path.exists(U.join(".", "requirements.txt")):
    with open(U.join(".", "requirements.txt"), "w") as f:
      f.write(f'nbox[serving]=={__version__}\n')

  # create a .nboxignore file if it doesn't exist, this will tell all the files not to be uploaded to the job pod
  _igp = set(ospec.FN_IGNORE)
  if not os.path.exists(U.join(".", ".nboxignore")):
    with open(U.join(".", ".nboxignore"), "w") as f:
      f.write("\n".join(_igp))
  else:
    with open(U.join(".", ".nboxignore"), "r") as f:
      _igp = _igp.union(set(f.read().splitlines()))
    _igp = sorted(list(_igp)) # sort it so it doesn't keep creating diffs in the git
    with open(U.join(".", ".nboxignore"), "w") as f:
      f.write("\n".join(_igp))

  # creation of resources, we first need to check if any resource arguments are passed, if they are
  def __common_resource(db: Resource) -> Resource:
    # get a common resource based on what the user has said, what the db has and defaults if nothing is given
    resource = Resource(
      cpu = str(resource_cpu) or db.cpu or ospec.DEFAULT_RESOURCE.cpu,
      memory = str(resource_memory) or db.memory or ospec.DEFAULT_RESOURCE.memory,
      disk_size = str(resource_disk_size) or db.disk_size or ospec.DEFAULT_RESOURCE.disk_size,
      gpu = str(resource_gpu) or db.gpu or ospec.DEFAULT_RESOURCE.gpu,
      gpu_count = str(resource_gpu_count) or db.gpu_count or ospec.DEFAULT_RESOURCE.gpu_count,
      timeout = int(resource_timeout) or db.timeout or ospec.DEFAULT_RESOURCE.timeout,
      max_retries = int(resource_max_retries) or db.max_retries or ospec.DEFAULT_RESOURCE.max_retries,
    )
    return resource

  # common to both, kept out here because these two will eventually merge
  nbx_auth_info = auth_info_pb()
  if method == ospec.OperatorType.JOB:
    # since user has not passed any arguments, we will need to check if the job already exists
    job_proto: JobProto = nbox_grpc_stub.GetJob(
      JobRequest(
        auth_info = nbx_auth_info,
        job = JobProto(id = id)
      )
    )
    resource = __common_resource(job_proto.resource)
    out: Job = deploy_job(
      init_folder = init_folder,
      job_id = job_proto.id,
      job_name = job_proto.name,
      dag = DAGProto(),
      workspace_id = workspace_id,
      schedule = None,
      resource = resource,
      exe_jinja_kwargs = exe_jinja_kwargs,
    )
    if trigger:
      logger.info(f"Triggering job: {job_proto.name} ({job_proto.id})")
      out = out.trigger()

  elif method == ospec.OperatorType.SERVING:
    model_name = model_name or U.get_random_name().replace("-", "_")
    logger.info(f"Model name: {model_name}")
    
    # serving_id, serving_name = _get_deployment_data(name = name, id = id, workspace_id = workspace_id)
    serving_proto: Serving = nbox_serving_service_stub.GetServing(
      ServingRequest(
        auth_info = nbx_auth_info,
        serving = Serving(id=id),
      )
    )
    resource = __common_resource(serving_proto.resource)
    out: Serve = deploy_serving(
      init_folder = init_folder,
      serving_id = serving_proto.id,
      model_name = model_name,
      serving_name = serving_proto.name,
      workspace_id = workspace_id,
      resource = resource,
      wait_for_deployment = False,
      model_metadata = {
        "serving_type": serving_type
      },
      exe_jinja_kwargs = exe_jinja_kwargs,
    )
    if trigger:
      out.pin()
  else:
    raise ValueError(f"Unknown method: {method}")

  os.chdir(_curdir)

  if _ret:
    return out


################################################################################
"""
# NimbleBox.ai Serving

This is the proposed interface for the NimbleBox.ai Serving API. We want to keep
the highest levels of consistency with the NBX-Jobs API.
"""
################################################################################

@lru_cache()
def _get_deployment_data(name: str = "", id: str = "", *, workspace_id: str = "") -> Tuple[str, str]:
  """
  Get the deployment data, either by name or id

  Args:
    name (str, optional): Name of the deployment. Defaults to "".
    id (str, optional): ID of the deployment. Defaults to "".

  Returns:
    Tuple[str, str]: (id, name)
  """
  # print("Getting deployment data", name, id, workspace_id)
  if (not name and not id) or (name and id):
    logger.warning("Must provide either name or id")
    return None, None
  # filter and get "id" and "name"
  workspace_id = workspace_id or secret(AuthConfig.workspace_id)

  # get the deployment
  serving: Serving = rpc(
    nbox_serving_service_stub.GetServing,
    ServingRequest(
      serving=Serving(name=name, id=id),
      auth_info = auth_info_pb()
    ),
    "Could not get deployment",
    raise_on_error=True
  )

  return serving.id, serving.name

def print_serving_list(sort: str = "created_on", *, workspace_id: str = ""):
  """
  Print the list of deployments

  Args:
    sort (str, optional): Sort by. Defaults to "created_on".
  """
  def _get_time(t):
    return datetime.fromtimestamp(int(float(t))).strftime("%Y-%m-%d %H:%M:%S")

  workspace_id = workspace_id or secret(AuthConfig.workspace_id)
  all_deployments: ServingListResponse = rpc(
    nbox_serving_service_stub.ListServings,
    ServingListRequest(
      auth_info=auth_info_pb(),
      limit=10
    ),
    "Could not get deployments",
    raise_on_error=True
  )
  # sorted_depls = sorted(all_deployments, key = lambda x: x[sort], reverse = sort == "created_on")
  # headers = ["created_on", "id", "name", "pinned_id", "pinned_name", "pinned_last_updated"]
  # [TODO] add sort by create time
  headers = ["id", "name", "pinned_id", "pinned_name", "pinned_last_updated"]
  all_depls = []
  for depl in all_deployments.servings:
    _depl = (depl.id, depl.name)
    pinned = depl.models[0] if len(depl.models) else None
    if not pinned:
      _depl += (None, None,)
    else:
      _depl += (pinned.id, pinned.name, _get_time(pinned.created_at.seconds))
    all_depls.append(_depl)

  for l in tabulate.tabulate(all_depls, headers).splitlines():
    logger.info(l)


class Serve:
  status = staticmethod(print_serving_list)
  upload: 'Serve' = staticmethod(partial(upload_job_folder, "serving"))

  def __init__(self, serving_id: str = "", model_id: str = "", *, workspace_id: str = "") -> None:
    """Python wrapper for NBX-Serving gRPC API

    Args:
      serving_id (str, optional): Serving ID. Defaults to None.
      model_id (str, optional): Model ID. Defaults to None.
    """
    self.id = serving_id
    self.model_id = model_id
    self.workspace_id = workspace_id or secret(AuthConfig.workspace_id)
    if workspace_id is None:
      raise DeprecationWarning("Personal workspace does not support serving")
    else:
      serving_id, serving_name = _get_deployment_data(name = "", id = self.id, workspace_id = self.workspace_id) # TODO add name support
    self.serving_id = serving_id
    self.serving_name = serving_name
    self.ws_stub = nbox_ws_v1.deployments

  def pin(self) -> bool:
    """Pin a model to the deployment

    Args:
      model_id (str, optional): Model ID. Defaults to None.
      workspace_id (str, optional): Workspace ID. Defaults to "".
    """
    logger.info(f"Pin model {self.model_id} to deployment {self.serving_id}")
    rpc(
      nbox_model_service_stub.SetModelPin,
      ModelRequest(
        model = ModelProto(
          id = self.model_id,
          serving_group_id = self.serving_id,
          pin_status = ModelProto.PinStatus.PIN_STATUS_PINNED
        ),
        auth_info = auth_info_pb()
      ),
      "Could not pin model",
      raise_on_error = True
    )
  
  def unpin(self) -> bool:
    """Pin a model to the deployment

    Args:
      model_id (str, optional): Model ID. Defaults to None.
      workspace_id (str, optional): Workspace ID. Defaults to "".
    """
    logger.info(f"Unpin model {self.model_id} to deployment {self.serving_id}")
    rpc(
      nbox_model_service_stub.SetModelPin,
      ModelRequest(
        model = ModelProto(
          id = self.model_id,
          serving_group_id = self.serving_id,
          pin_status = ModelProto.PinStatus.PIN_STATUS_UNPINNED
        ),
        auth_info = auth_info_pb(),
      ),
      "Could not unpin model",
      raise_on_error = True
    )
  
  def scale(self, replicas: int) -> bool:
    """Scale the model deployment

    Args:
      replicas (int): Number of replicas
    """
    if not self.model_id:
      raise ValueError("Model ID is required to scale a model. Pass with --model_id")
    if replicas < 0:
      raise ValueError("Replicas must be greater than or equal to 0")

    logger.info(f"Scale model deployment {self.model_id} to {replicas} replicas")
    rpc(
      nbox_model_service_stub.UpdateModel,
      UpdateModelRequest(
        model=ModelProto(
          id=self.model_id,
          serving_group_id=self.serving_id,
          replicas=replicas
        ),
        update_mask=FieldMask(paths=["replicas"]),
        auth_info = auth_info_pb()
      ),
      "Could not scale deployment",
      raise_on_error = True
    )
  
  def logs(self, f = sys.stdout):
    """Get the logs of the model deployment

    Args:
      f (file, optional): File to write the logs to. Defaults to sys.stdout.
    """
    logger.debug(f"Streaming logs of model '{self.model_id}'")
    for model_log in streaming_rpc(
      nbox_model_service_stub.ModelLogs,
      ModelRequest(
        model = ModelProto(
          id = self.model_id,
          serving_group_id = self.serving_id
        ),
        auth_info = auth_info_pb(),
      ),
      f"Could not get logs of model {self.model_id}, only live logs are available",
      False
    ):
      for log in model_log.log:
        f.write(log)
        f.flush()

  def __repr__(self) -> str:
    x = f"nbox.Serve('{self.id}', '{self.workspace_id}'"
    if self.model_id is not None:
      x += f", model_id = '{self.model_id}'"
    x += ")"
    return x



################################################################################
"""
# NimbleBox.ai Jobs

This is the actual job object that users can manipulate. It is a shallow class
around the NBX-Jobs gRPC API.
"""
################################################################################

@lru_cache()
def _get_job_data(name: str = "", id: str = "", remove_archived: bool = True, *, workspace_id: str = "") -> Tuple[str, str]:
  """
  Returns the job id and name
  
  Args:
    name (str, optional): Job name. Defaults to "".
    id (str, optional): Job ID. Defaults to "".
    remove_archived (bool, optional): If True, will remove archived jobs. Defaults to True.

  Returns:
    Tuple[str, str]: Job ID, Job Name
  """
  if (not name and not id) or (name and id):
    logger.info(f"Please either pass job_id '{id}' or name '{name}'")
    return None, None
  # get stub
  workspace_id = workspace_id or secret(AuthConfig.workspace_id)
  if workspace_id == None:
    workspace_id = "personal"

  job: JobProto = rpc(
    nbox_grpc_stub.GetJob,
    JobRequest(
      auth_info = auth_info_pb(),
      job = JobProto(id=id, name=name)
    ),
    "Could not find job with ID: {}".format(id),
    raise_on_error = True
  )
  if job.status == JobProto.Status.ARCHIVED and remove_archived:
    logger.info(f"Job {job.id} is archived. Will try to create a new Job.")
    return None, name
  job_name = job.name
  job_id = job.id
  logger.info(f"Found job with ID '{job_id}' and name '{job_name}'")
  return job_id, job_name

def get_job_list(sort: str = "name", *, workspace_id: str = ""):
  """Get list of jobs
  
  Args:
    sort (str, optional): Sort key. Defaults to "name".
  """
  workspace_id = workspace_id or secret(AuthConfig.workspace_id)

  def _get_time(t):
    return datetime.fromtimestamp(int(float(t))).strftime("%Y-%m-%d %H:%M:%S")

  out: ListJobsResponse = rpc(
    nbox_grpc_stub.ListJobs,
    ListJobsRequest(auth_info = auth_info_pb()),
    "Could not get job list",
  )

  if len(out.jobs) == 0:
    logger.info("No jobs found")
    sys.exit(0)

  headers = ['created_at', 'id', 'name', 'schedule', 'status']
  try:
    sorted_jobs = sorted(out.jobs, key = lambda x: getattr(x, sort))
  except Exception as e:
    logger.error(f"Cannot sort on key: {sort}")
    sorted_jobs = out.jobs
  data = []
  for j in sorted_jobs:
    _row = []
    for x in headers:
      if x == "status":
        _row.append(JobProto.Status.keys()[getattr(j, x)])
        continue
      elif x == "created_at":
        _row.append(_get_time(j.created_at.seconds))
      elif x == "schedule":
        _row.append(j.schedule.cron)
      else:
        _row.append(getattr(j, x))
    data.append(_row)
  for l in tabulate.tabulate(data, headers).splitlines():
    logger.info(l)


class Job:
  status = staticmethod(get_job_list)
  upload: 'Job' = staticmethod(partial(upload_job_folder, "job"))

  def __init__(self, job_name: str = "", job_id: str = ""):
    """Python wrapper for NBX-Jobs gRPC API, when both arguments are not passed,
    an unintialiased object is created.

    Args:
      job_name (str, optional): Job name. Defaults to "".
      job_id (str, optional): Job ID. Defaults to "".
    """
    if job_name == "" and job_id == "":
      return

    self.id, self.name = _get_job_data(job_name, job_id)
    self.workspace_id = secret(AuthConfig.workspace_id)
    self.auth_info = auth_info_pb()
    self.job_proto = JobProto(id = self.id)

    self.run_stub = None
    self.runs = []

    # sometimes a Job can be a placeholder and not have any data against it, thus it won't make
    # sense to try to get the data. This causes RPC error
    if self.id is not None:
      self.refresh()
      self.get_runs() # load the runs as well

  @property
  def exists(self):
    """Check if this job exists in the workspace"""
    return self.id is not None

  @classmethod
  def new(cls, job_name: str, description: str = ""):
    """Create a new job

    Args:
      job_name (str): Job name
      description (str, optional): Job description. Defaults to "".

    Returns:
      Job: Job object
    """
    job = nbox_ws_v1.job("post", job_name = job_name, job_description = description)
    job_id = job["job_id"]
    # job_proto: JobProto = nbox_grpc_stub.CreateJob(
    #   JobRequest(
    #     auth_info = auth_info_pb(),
    #     job = JobProto(
    #       name = job_name,
    #       code = Code(
    #         size = 1,
    #         type = Code.Type.NOT_SET,
    #       ),
    #       resource = DEFAULT_RESOURCE,
    #       schedule = JobProto.Schedule(
    #         cron = "0 0 * * *",
    #         end = Timestamp(seconds = 0),
    #       ),
    #       status = JobProto.Status.NOT_SET,
    #       paused = False,
    #     ),
    #   )
    # )
    logger.info(f"Created a new job with name (ID): {job_name} ({job_id})")
    return cls(job_id = job_id)

  def change_schedule(self, new_schedule: Schedule):
    """Change schedule this job
    
    Args:
      new_schedule (Schedule): New schedule
    """
    logger.debug(f"Updating job '{self.job_proto.id}'")
    self.job_proto.schedule.MergeFrom(new_schedule.get_message())
    rpc(
      nbox_grpc_stub.UpdateJob,
      UpdateJobRequest(auth_info=self.auth_info, job=self.job_proto, update_mask=FieldMask(paths=["schedule"])),
      "Could not update job schedule",
      raise_on_error = True
    )
    logger.debug(f"Updated job '{self.job_proto.id}'")
    self.refresh()

  def __repr__(self) -> str:
    x = f"nbox.Job('{self.job_proto.id}', '{self.job_proto.auth_info.workspace_id}'): {self.status}"
    if self.job_proto.schedule.ByteSize != None:
      x += f" {self.job_proto.schedule}"
    else:
      x += " (no schedule)"
    return x

  def logs(self, f = sys.stdout):
    """Stream logs of the job, `f` can be anything has a `.write/.flush` methods"""
    logger.debug(f"Streaming logs of job '{self.job_proto.id}'")
    for job_log in streaming_rpc(
      nbox_grpc_stub.GetJobLogs,
      JobRequest(auth_info=self.auth_info ,job = self.job_proto),
      f"Could not get logs of job {self.job_proto.id}, is your job complete?",
      True
    ):
      for log in job_log.log:
        f.write(log)
        f.flush()

  def delete(self):
    """Delete this job"""
    logger.info(f"Deleting job '{self.job_proto.id}'")
    rpc(nbox_grpc_stub.DeleteJob, JobRequest(auth_info=self.auth_info, job = self.job_proto,), "Could not delete job")
    logger.info(f"Deleted job '{self.job_proto.id}'")
    self.refresh()

  def refresh(self):
    """Refresh Job data"""
    logger.info(f"Updating job '{self.job_proto.id}'")
    if self.id == None:
      self.id, self.name = _get_job_data(id = self.id, workspace_id = self.workspace_id)
    if self.id == None:
      return

    self.job_proto: JobProto = rpc(
      nbox_grpc_stub.GetJob,
      JobRequest(auth_info=self.auth_info, job = self.job_proto),
      f"Could not get job {self.job_proto.id}"
    )
    self.auth_info.CopyFrom(auth_info_pb())
    logger.debug(f"Updated job '{self.job_proto.id}'")

    self.status = self.job_proto.Status.keys()[self.job_proto.status]

  def trigger(self, tag: str = ""):
    """Manually triger this job.
    
    Args:
      tag (str, optional): Tag to be set in the run metadata, read in more detail before trying to use this. Defaults to "".
    """
    logger.debug(f"Triggering job '{self.job_proto.id}'")
    if tag:
      self.job_proto.feature_gates.update({"SetRunMetadata": tag})
    rpc(nbox_grpc_stub.TriggerJob, JobRequest(auth_info=self.auth_info, job = self.job_proto), f"Could not trigger job '{self.job_proto.id}'")
    logger.info(f"Triggered job '{self.job_proto.id}'")
    self.refresh()

  def pause(self):
    """Pause the execution of this job.

    **WARNING: This will "cancel" all the scheduled runs, if present**
    """
    logger.info(f"Pausing job '{self.job_proto.id}'")
    job: JobProto = self.job_proto
    job.status = JobProto.Status.PAUSED
    rpc(nbox_grpc_stub.UpdateJob, UpdateJobRequest(auth_info=self.auth_info, job=job, update_mask=FieldMask(paths=["status", "paused"])), f"Could not pause job {self.job_proto.id}", True)
    logger.debug(f"Paused job '{self.job_proto.id}'")
    self.refresh()

  def resume(self):
    """Resume the Job with the current schedule, if provided else simlpy sets status as ACTIVE"""
    logger.info(f"Resuming job '{self.job_proto.id}'")
    job: JobProto = self.job_proto
    job.status = JobProto.Status.SCHEDULED
    rpc(nbox_grpc_stub.UpdateJob, UpdateJobRequest(auth_info=self.auth_info, job=job, update_mask=FieldMask(paths=["status", "paused"])), f"Could not resume job {self.job_proto.id}", True)
    logger.debug(f"Resumed job '{self.job_proto.id}'")
    self.refresh()

  def _get_runs(self, page = -1, limit = 10) -> List[Dict]:
    self.run_stub = nbox_ws_v1.job.u(self.id).runs
    runs = self.run_stub(limit = limit, page = page)["runs_list"]
    return runs

  def get_runs(self, page = -1, sort = "s_no", limit = 10) -> List[Dict]:
    """
    Get runs for this job

    Args:
      page (int, optional): Page number. Defaults to -1.
      sort (str, optional): Sort by. Defaults to "s_no".
      limit (int, optional): Number of runs to return. Defaults to 10.

    Returns:
      List[Dict]: List of runs
    """
    runs = self._get_runs(page, limit)
    sorted_runs = sorted(runs, key = lambda x: x[sort])
    return sorted_runs

  def display_runs(self, sort: str = "created_at", page: int = -1, limit = 10):
    """Display runs for this job

    Args:
      sort (str, optional): Sort by. Defaults to "created_at".
      page (int, optional): Page number. Defaults to -1.
      limit (int, optional): Number of runs to return. Defaults to 10.
    """
    headers = ["s_no", "created_at", "run_id", "status"]

    def _display_runs(runs):
      data = []
      for run in runs:
        data.append((run["s_no"], run["created_at"], run["run_id"], run["status"]))
      for l in tabulate.tabulate(data, headers).splitlines():
        logger.info(l)

    runs = self.get_runs(sort = sort, page = page, limit = limit) # time should be reverse
    _display_runs(runs)
    if page == -1:
      page = 1
    y = input(f">> Print {limit} more runs? (y/n): ")
    done = y != "y"
    while not done:
      _display_runs(self.get_runs(page = page + 1, sort = sort, limit = limit))
      page += 1
      y = input(f">> Print {limit} more runs? (y/n): ")
      done = y != "y"

  def last_n_runs(self, n: int = 10) -> List[Dict]:
    """Get last n runs for this job

    Args:
      n (int, optional): Number of runs to return. Defaults to 10.

    Returns:
      List[Dict]: List of runs
    """
    all_items = []
    _page = 1
    out = self._get_runs(_page, n)
    all_items.extend(out)

    while len(all_items) < n:
      _page += 1
      out = self._get_runs(_page, n)
      if not len(out):
        break
      all_items.extend(out)

    if n == 1:
      return all_items[0]
    return all_items
