import os

from nbox import utils as U
from nbox import messages as mpb
from nbox.utils import logger, lo
from nbox.auth import secret, auth_info_pb
from nbox.version import __version__
from nbox.init import nbox_grpc_stub, nbox_serving_service_stub

from nbox.nbxlib.astea import Astea, IndexTypes as IT
from nbox.hyperloop.common.common_pb2 import Resource
from nbox.hyperloop.jobs.job_pb2 import (
  Job as JobProto,
)
from nbox.hyperloop.jobs.dag_pb2 import DAG as DAGProto
from nbox.hyperloop.jobs.nbox_ws_pb2 import JobRequest
from nbox.hyperloop.deploy.serve_pb2 import (
  ServingRequest,
  Serving,
)

from nbox.jd_core.jobs import Job
from nbox.jd_core.serving import Serve


def upload_job_folder(
  method: str,
  init_folder: str,
  id: str = "",
  project_id: str = "",

  # job / deploy rpc things
  trigger: bool = False,
  deploy: bool = True,
  pin: bool = False,
  feature_gates: dict = {},

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
  if trigger and method != OT.JOB:
    raise ValueError(f"Trigger can only be used with method='{OT.JOB}'")
  if pin and method != OT.SERVING:
    raise ValueError(f"Deploy and Pin can only be used with method='{OT.SERVING}'")
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

  workspace_id = workspace_id or secret.workspace_id

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
    logger.debug(lo("db resources", **mpb.message_to_dict(db)))
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
    logger.debug(lo("resources", **mpb.message_to_dict(resource)))
    out: Job = deploy_job(
      init_folder = init_folder,
      job_id = job_proto.id,
      job_name = job_proto.name,
      feature_gates = feature_gates,
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
    logger.debug(lo("resources", **mpb.message_to_dict(resource)))
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
    if deploy:
      out.deploy(feature_gates = feature_gates, resource = resource)
    if trigger:
      out.pin()
  else:
    raise ValueError(f"Unknown method: {method}")

  os.chdir(_curdir)

  if _ret:
    return out