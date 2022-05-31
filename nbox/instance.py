"""
NBX-Build Instances are APIs to your machines. These APIs can be used to change state of
the machine (start, stop, etc.), can be used to transfer files to and from the machine
and to an extent running and managing programs (WIP).
"""

import os
import sys
import time
import shlex
from typing import List
from functools import partial
from tabulate import tabulate
from tempfile import gettempdir
from subprocess import PIPE, run
from requests.sessions import Session

from .subway import SpecSubway,  TIMEOUT_CALLS
from . import utils as U
from .utils import NBOX_HOME_DIR, logger
from .init import nbox_ws_v1, create_webserver_subway
from .auth import secret


################################################################################
'''
# NBX-Instances Functions

The methods below are used to talk to the Webserver APIs and other methods
make the entire process functional.
'''
################################################################################

def print_status(workspace_id: str = None, fields: List[str] = None):
  """Print complete status of NBX-Build instances. If ``workspace_id`` is not provided
  personal workspace will be used. Used in CLI"""
  logger.info("Getting NBX-Build details")
  if workspace_id == None:
    stub_projects = nbox_ws_v1.user.projects
  else:
    stub_projects = nbox_ws_v1.workspace.u(workspace_id).projects

  fields = fields if fields != None else Instance.useful_keys

  data = stub_projects()["data"]
  projects = data["project_details"]
  data = [{k: projects[x][k] for k in fields} for x in projects]
  data_table = [[x[k] for k in fields] for x in data]
  for x in tabulate(data_table, headers=fields).splitlines():
    logger.info(x)

################################################################################
'''
# NimbleBox Build

NBX-Instances is compute abstracted away to get the best hardware for your
task. To that end each Instance from the platform is a single class.
'''
################################################################################

class Instance():
  status = staticmethod(print_status)

  # each instance has a lot of data against it, we need to store only a few as attributes
  useful_keys = ["project_id", "project_name", "size_used", "size", "state",]

  def __init__(self, i: str, workspace_id: int = None, cs_endpoint = "server"):
    """NBX-Build Instance class manages the both individual instance, but provides
    webserver functionality using ``nbox_ws_v1``, such as starting and stopping,
    deletion and more.

    Args:
        i (str): name or ``project_id`` of the instance
        workspace_id (int, optional): id of the workspace to use, if not provided
            personal workspace will be used.
        cs_endpoint (str, optional): endpoint to use for the webserver, this will connect
          to the custom ports functionality of the instance. Defaults to ``server``, 
    """
    if i == None:
      raise ValueError("Instance id must be provided, try --i='8h57f9'")
    i = str(i)

    # simply add useful keys to the instance
    self.project_id: str = None
    self.project_name: str = None
    self.workspace_id: str = workspace_id
    self.size_used: float = None
    self.size: float = None
    self.state: str = None

    # create a new session for communication with the compute server, avoid using single
    # session for conlficting headers
    sess = Session()
    sess.headers.update({"Authorization": f"Bearer {secret.get('access_token')}"})
    stub_ws_instance = create_webserver_subway("v1", sess)

    self.status = None # this is the status string
    if workspace_id == None:
      stub_projects = stub_ws_instance.user.projects
    else:
      stub_projects = stub_ws_instance.workspace.u(workspace_id).projects

    project_details = stub_projects()["data"]["project_details"]
    if i not in project_details:
      by_name = list(filter(lambda x: x['project_name'] == i, list(project_details.values())))
      if len(by_name) == 0:
        raise ValueError(f"Instance '{i}' not found")
      elif len(by_name) > 1:
        raise ValueError(f"Multiple instances with name '{i}' found")
      data = by_name[0]
    else:
      data = project_details[i]
    logger.info(f"Found instance '{data['project_name']}' ({data['project_id']})")

    for x in self.useful_keys:
      setattr(self, x, data[x])

    self.url = secret.get('nbx_url')
    self.cs_url = None
    self.cs_endpoint = cs_endpoint

    self.stub_ws_instance = stub_projects.u(self.project_id) # .../build/{project_id}
    logger.debug(f"WS: {self.stub_ws_instance}")
    self.__opened = False
    self.running_scripts = []
    self.refresh()
    if self.state == "RUNNING":
      self.start()
    logger.debug(f"Instance: {self}")

  def __repr__(self):
    return f"<Instance ({', '.join([f'{k}:{getattr(self, k)}' for k in self.useful_keys + ['cs_url']])})>"

  """
  # Classmethods (WIP)

  Functions to create instances.
  """

  @classmethod
  def new(cls, project_name: str, workspace_id: str = None, storage_limit: int = 25, project_type = "blank") -> 'Instance':
    if workspace_id == None:
      stub_all_projects = nbox_ws_v1.user.projects
    else:
      stub_all_projects = nbox_ws_v1.workspace.u(workspace_id).projects
    out = stub_all_projects(
      _method = "post",
      project_name = project_name,
      storage_limit = storage_limit,
      project_type = project_type,
      github_branch = "",
      github_link = "",
      template_id = 0,
      clone_id = 0
  )
    
    return out

  ################################################################################
  """
  # State Methods

  Functions here are responsible for managing the states (metadata + turn on/off) of the instances.
  """
  ################################################################################

  def is_running(self) -> bool:
    """Check if the instance is running.

    Returns:
        bool: True if the instance is running, False otherwise.
    """
    return self.state == "RUNNING"

  def refresh(self):
    """Update the data, get latest state"""
    self.data = self.stub_ws_instance()["data"] # GET /user/projects/{project_id}
    for k in self.useful_keys:
      setattr(self, k, self.data[k])

  def _start(self, cpu, gpu, gpu_count, auto_shutdown, dedicated_hw, zone):
    """Turn on the the unserlying compute"""
    logger.info(f"Starting instance {self.project_name} ({self.project_id})")
    hw_config = {
      "cpu":f"n1-standard-{cpu}"
    }
    if gpu_count > 0:
      hw_config["gpu"] = f"nvidia-tesla-{gpu}"
      hw_config["gpu_count"] = gpu_count
    else:
      # things nimblebox does, for nimblebox reasons
      hw_config["gpu"] = 'null'

    self.stub_ws_instance.start(
      auto_shutdown = auto_shutdown == 0,
      auto_shutdown_value = auto_shutdown,
      dedicated_hw = dedicated_hw,
      hw = "cpu" if gpu_count == 0 else "gpu",
      hw_config = hw_config,
      zone = zone
    )

    logger.info(f"Waiting for instance {self.project_name} ({self.project_id}) to start ...")
    _i = 0
    while self.state != "RUNNING":
      time.sleep(5)
      self.refresh()
      _i += 1
      if _i > TIMEOUT_CALLS:
        raise TimeoutError("Instance did not start within timeout, please check dashboard")
    logger.info(f"Instance {self.project_name} ({self.project_id}) started")

  def start(
    self,
    cpu: int = 2,
    gpu: str = "p100",
    gpu_count: int = 0,
    auto_shutdown: int = 6,
    dedicated_hw: bool = False,
    zone = "asia-south-1",
    *,
    _ssh: bool = False
  ):
    """Start instance if not already running and loads APIs from the compute server.

    Actual start is implemented in `_start` method, this combines other things 

    Args:
        cpu (int, optional): CPU count should be one of ``[2, 4, 8]``
        gpu (str, optional): GPU name should be one of ``["t5", "p100", "v100", "k80"]``
        gpu_count (int, optional): When zero, cpu-only instance is started
        auto_shutdown (int, optional): No autoshutdown if zero, defaults to 6.
        dedicated_hw (bool, optional): If not spot/pre-emptible like machines used
        zone (str, optional): GCP cloud regions, defaults to "asia-south-1".
    """
    if _ssh:
      if "app.nimblebox.ai" in self.stub_ws_instance._url:
        self.stub_ws_instance._url = self.stub_ws_instance._url.replace("app.", "app.rc.")

    if auto_shutdown < 0:
      raise ValueError("auto_shutdown must be a positive integer (hours)")
    gpu_count = int(gpu_count)

    if not self.state == "RUNNING":
      self._start(cpu, gpu, gpu_count, auto_shutdown, dedicated_hw, zone)
    else:
      # TODO: @yashbonde: inform user in case of hardware mismatch?
      logger.info(f"Instance {self.project_name} ({self.project_id}) is already running")

    # now the instance is running, we can open it, opening will assign a bunch of cookies and
    # then get us the exact location of the instance
    logger.debug(f"Opening instance {self.project_name} ({self.project_id})")
    base_domain = self.stub_ws_instance.launch(_method = "post")["data"]["base_domain"]
    secret.get("url")
    self.open_data = {
      "url": f"{base_domain}",
      "token": self.stub_ws_instance._session.cookies.get_dict()[f"instance_token_{base_domain}"]
    }

    instance_url = self.open_data["url"].strip("/")
    self.cs_url = f"{self.url}/{instance_url}"
    if self.cs_endpoint:
      self.cs_url += f"/{self.cs_endpoint}"

    # TODO:@yashbonde remove this check, once v2 on prod
    if self.cs_url.endswith("server"):
      self.__opened = True
      return

    # create a speced-subway, this requires capturing the openAPI spec first
    r = self.session.get(f"{self.cs_url}/openapi.json"); r.raise_for_status()
    self.cs_spec = r.json()
    self.compute_server = SpecSubway.from_openapi(self.cs_spec, self.cs_url, self.session)
    logger.debug(f"CS: {self.compute_server}")

    # now load all the functions from methods.py
    logger.debug(f"Testing instance {self.project_name} ({self.project_id})")
    out = self.compute_server.test()
    with open(U.join(NBOX_HOME_DIR, "methods.py"), "w") as f:
      f.write(out["data"])

    self.__opened = True

  def stop(self):
    """Stop Instance"""
    if self.state == "STOPPED":
      logger.info(f"Instance {self.project_name} ({self.project_id}) is already stopped")
      return

    logger.debug(f"Stopping instance {self.project_name} ({self.project_id})")
    message = self.stub_ws_instance.stop_instance("post", data = {"instance_id":self.project_id})["msg"]
    if not message == "success":
      raise ValueError(message)

    logger.debug(f"Waiting for instance {self.project_name} ({self.project_id}) to stop")
    _i = 0 # timeout call counter
    while self.state != "STOPPED":
      time.sleep(5)
      self.refresh()
      _i += 1
      if _i > TIMEOUT_CALLS:
        raise TimeoutError("Instance did not stop within timeout, please check dashboard")
    logger.debug(f"Instance {self.project_name} ({self.project_id}) stopped")

    self.__opened = False

  def delete(self, force = False):
    """With great power comes great responsibility."""
    if self.__opened and not force:
      raise ValueError("Instance is still opened, please call .stop() first")
    logger.warning(f"Deleting instance {self.project_name} ({self.project_id})")
    if input(f"> Are you sure you want to delete '{self.project_name}'? (y/N): ") == "y":
      self.stub_ws_instance("delete")
    else:
      logger.warning("Aborted")

  ################################################################################
  """
  # Interaction Methods

  Doing things with the project-build.
  """
  ################################################################################

  def __unopened_error(self):
    if not self.__opened:
      logger.error("You are trying to move files to a NOT-RUNNING instance, you will have to start the instance first:")
      logger.error('    -         nbox.Instance(...).start(...)')
      logger.error('    - python3 -m nbox build ... start ...')
      raise ValueError("Instance is not opened, please call .open() first")

  def __run_command(self, comm: str, port: int) -> str:
    from nbox.sub_utils.ssh import _create_threads
    connection = _create_threads(port, i = self.project_id, workspace_id = self.workspace_id, _ssh = False)
    try:
      command = shlex.split(comm)
      logger.info(f"Running command: {comm}")
      result = run(command, stdout=PIPE, stderr=PIPE, universal_newlines=True)
      result = result.stdout
    except KeyboardInterrupt:
      logger.info("KeyboardInterrupt, closing connections")
      result = ""
    connection.quit()
    return result

  def ls(self, path: str, *, port: int = 6174):
    """
    List files in a directory relative to '/home/ubuntu/project'

    ```
    [nbox API] - nbox.Instance(...).ls("/)
    [nbox CLI] - python3 -m nbox build ... ls /
    ```
    """
    self.__unopened_error()

    if path == "":
      raise ValueError("Path cannot be empty")
    path = "/home/ubuntu/project/" + path.strip("/")

    # RPC
    logger.info(f"Looking in folder: {path}")
    result = self.__run_command(f"ssh -p {port} ubuntu@localhost 'ls -l {path}'", port)
    return result

  def mv(self, src: str, dst: str, force: bool = False, *, port: int = 6174):
    """
    Move files to and fro NBX-Build.
    
    Use 'nbx://' as prefix for Instance, all files will be placed relative to
    "/home/ubuntu/project/" folder.
    """
    self.__unopened_error()

    src_is_cloud = src.startswith("nbx://")
    dst_is_cloud = dst.startswith("nbx://")

    if src_is_cloud and dst_is_cloud:
      raise ValueError("Cannot move files between two instances")
    if not src_is_cloud and not dst_is_cloud:
      raise ValueError("Cannot move files on your machine")
    elif src_is_cloud:
      if os.path.exists(dst) and not force:
        raise ValueError(f"Source file '{dst}' already exists, pass force=True to overwrite")
    elif dst_is_cloud:
      assert os.path.exists(src), f"Source file '{src}' does not exist"
    
    # check if this file already exists
    cloud_file = src if src_is_cloud else dst
    cloud_file = cloud_file.replace("nbx://", "")

    ls_res = self.ls(cloud_file)
    logger.info(f"ls_res: {ls_res}")
    if ls_res:
      logger.info(f"File {cloud_file} already exists")
      if not force:
        logger.error("Aborted, use --force to override")
        return
      logger.info("Overriding file")

    src = "/home/ubuntu/project/" + src[6:] if src_is_cloud else src
    dst = "/home/ubuntu/project/" + dst[6:] if dst_is_cloud else dst

    src_folder = os.path.isdir(src) if not src_is_cloud else os.path.isdir(src)
    src_folder = "-r" if src_folder else ""

    # RPC
    logger.info(f"Moving {src} to {dst}")
    result = self.__run_command(f'scp {src_folder} -P {port} {src} localhost:{dst}', port)
    return result

  def rm(self, file: str, *, port: int = 6174):
    """
    Remove file from NBX-Build.
    """
    self.__unopened_error()

    if not file.startswith("nbx://"):
      raise ValueError("File must be on NBX-Build, try adding 'nbx://' to start!")
    file = "/home/ubuntu/project/" + file.replace("nbx://", "")

    # RPC
    logger.info(f"Removing file: {file}")
    result = self.__run_command(f"ssh -p {port} ubuntu@localhost 'rm {file}'", port)
    return result

  def remote(self, x: str, *, port: int = 6174):
    """
    Run any command using SSH, in the underlying system it will setup a SSH connection and execute
    any command.
    """
    self.__unopened_error()

    # RPC
    logger.info(f"Running command: {x}")
    result = self.__run_command(f"ssh -p {port} ubuntu@localhost '{x}'", port)
    return result

  def __call__(self, x: str):
    """EXPERIMENTAL: FEATURES MIGHT BREAK
    
    Caller is the most important UI/UX. The letter ``x`` in programming is reserved the most
    arbitrary thing, and this ``nbox.Instance`` is the gateway to a cloud instance. You can:
    1. run a script on the cloud
    2. run a local script on the cloud
    3. get the status of a script
    4. [TBD] run a special kind of functions known as
    `pure functions <https://en.wikipedia.org/wiki/Pure_function>`_

    Pure functions in programming are functions that are self sufficient in terms of execution,
    eg. all the packages are imported inside the function and there are no side effects in an
    execution (seeds included for probabilistic functions). Writing such functions in python with
    any IDE with checker is dead easy, however the performace guarantee is premium given high
    costs. Thus the logic to parsing these will have to be written as a seperate module.
    """
    self.__unopened_error()
    raise NotImplementedError("Core talking method, not implemented yet")

    if not self.__opened:
      raise ValueError("Instance is not opened, please call .start() first")
    if not isinstance(x, str):
      raise ValueError("x must be a string")

      # this is to check if the PID state, returns boolean
    if x.startswith("PID_"):
      x = x[4:]
      data = self.compute_server.rpc.status(x)
      if not data["msg"] == "success":
        raise ValueError(data["msg"])
      
      status = data["status"]
      logger.debug(f"Script {x} on instance {self.project_name} ({self.project_id}) is {status}")

      if status in ["stopped", "failed"]:
        out = self.compute_server.rpc.logs(x)
        if out["err"]:
          logger.error(f"PID: {x} Error")
          print(out)
          return "error-done"
        else:
          return "done"
      return status

    # # assuming this is a shell command
    # command = shlex.join(shlex.split(x)) # sanitize the input
    comm_hash = U.hash_(x)
    # sh_fpath = U.join(gettempdir(), "run.sh")
    # with open(sh_fpath, "w") as f:
    #   f.write(command)
    # logger.info(f"Running command '{command}' [{comm_hash}] on instance {self.project_name} ({self.project_id})")
    # nbx_fpath = f"nbx://{comm_hash}.sh"
    # fpath = nbx_fpath.replace("nbx://", "/mnt/disks/user/project/")
    # self.mv(sh_fpath, nbx_fpath)
    # meta = list(filter(
    #   lambda y: y["path"] == fpath, self.compute_server.myfiles.info(fpath)["data"]
    # ))
    # if not meta:
    #   raise ValueError(f"File {x} not found on instance {self.project_name} ({self.project_id})")
    uid = "PID_" + self.compute_server.rpc.start(x, args = ["run"])["uid"]
    logger.info(f"Command {comm_hash} is running with PID {uid}")
    self.running_scripts.append(uid)
    return uid
