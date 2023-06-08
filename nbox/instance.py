"""
NBX-Build Instances are APIs to your machines. These APIs can be used to change state of the machine (start, stop,
etc.), can be used to transfer files to and from the machine (using `Instance.mv` commands), calling any shell
command using `Instance.remote`.

### CLI Commands

Build comes built in with several functions to move and list files from the instance. All you need to know is that
cloud files have prefix `nbx://` which is where your projects are. Here's a quick list:

```bash

# move files
nbx build -i 'instance_name' --workspace_id 'workspace_id' \
  mv ./local_file nbx://cloud_file

# or move folders
nbx build -i 'instance_name' --workspace_id 'workspace_id' \
  mv ./local_folder nbx://in_this/folder/

```

You might be required to agree to the SSH connection being setup. If you want to avoid that set `NBOX_SSH_NO_HOST_CHECKING=1`.
All these APIs are also available in python.

"""

import io
import os
import time
import shlex
from json import loads
from subprocess import run, check_output
from tabulate import tabulate
from typing import Dict, List
from requests.sessions import Session

import nbox.utils as U
from nbox.auth import secret, AuthConfig
from nbox.utils import logger
from nbox.subway import TIMEOUT_CALLS, Sub30
from nbox.init import nbox_ws_v1, create_webserver_subway


################################################################################
'''
# NBX-Instances Functions

The methods below are used to talk to the Webserver APIs and other methods
make the entire process functional.
'''
################################################################################

def print_status(fields: List[str] = [], *, workspace_id: str = ""):
  """Print complete status of NBX-Build instances. If `workspace_id` is not provided
  personal workspace will be used. Used in CLI
  
  Args:
    fields (List[str], optional): fields to print. Defaults to []. If not provided all fields will be printed.
  """
  logger.info("Getting NBX-Build details")
  stub_projects = nbox_ws_v1.projects

  fields = fields or Instance.useful_keys

  data = stub_projects()
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
  useful_keys = ["instance_id", "project_name", "size_used", "size", "state"]

  def __init__(self, i: str, *, workspace_id: str = ""):
    """NBX-Build Instance class manages the both individual instance, but provides webserver functionality using
    `nbox_ws_v1`, such as starting and stopping, deletion and more.

    Args:
      i (str): name or `instance_id` of the instance
    """
    if not i:
      raise ValueError("Instance id must be provided, try --i='1023'")
    
    # if user provided a number we assume that they gave an instance ID, this is a weak assumption because
    # people usually use names.
    _instance_id = isinstance(i, int)
    i = str(i)

    # simply add useful keys to the instance
    self.instance_id: str = None
    self.project_name: str = None
    self.workspace_id: str = workspace_id or secret.workspace_id
    self.size_used: float = None
    self.size: float = None
    self.state: str = None

    # few states are defined for use in the class
    self.__opened = False

    # create a new subway for webserver
    sess = Session()
    sess.headers.update({"Authorization": f"Bearer {secret.access_token}"})
    stub_ws_instance = create_webserver_subway("v1", sess)
    stub_projects = stub_ws_instance.instances

    if _instance_id:
      # if user provided an instance id, we can directly get the data
      data = stub_projects.u(i)()
      instance_id = i
    else:
      # else filter and get the data
      project_details = stub_projects()["project_details"]
      if i not in project_details:
        by_name = list(filter(lambda x: x[1]['project_name'] == i, list(project_details.items())))
        if len(by_name) == 0:
          raise ValueError(f"Instance '{i}' not found")
        elif len(by_name) > 1:
          raise ValueError(f"Multiple instances with name '{i}' found")
        data = by_name[0]
        instance_id = data[0]
        data = data[1]
      else:
        data = project_details[i]
        instance_id = i

    data["instance_id"] = instance_id
    logger.info(f"Found instance '{data['project_name']}' ({data['instance_id']})")
    # print(data)
    for x in Instance.useful_keys:
      setattr(self, x, data[x])
    
    # some data points require extra processing before usage
    self.custom_ports: Dict[str, int] = loads(data["custom_ports"]) if data["custom_ports"] is not None else {}
    self.exposed_ports: Dict[str, int] = loads(data["exposed_ports"]) if data["exposed_ports"] is not None else {}
    self.stub_ws_instance = stub_projects.u(self.instance_id)
    logger.debug(f"WS: {self.stub_ws_instance}")

    # set values
    self.refresh()
    if self.state == "RUNNING":
      self._open()

  def __repr__(self):
    return f"<Instance ({', '.join([f'{k}:{getattr(self, k)}' for k in self.useful_keys])})>"

  ################################################################################
  """
  # Classmethods

  Functions to create instances.
  """
  ################################################################################

  @classmethod
  def new(
    cls,
    project_name: str,
    storage_limit: int = 25,
    project_type: str = "blank",
    github_branch: str = "",
    github_link: str = "",
    template_id: int = 0,
    clone_id: int = 0,
    *,
    workspace_id: str = "",
  ) -> 'Instance':
    """Create a new NBX-Build instance.
    
    Args:
      project_name (str): Name of the instance
      storage_limit (int, optional): Storage limit in GB. Defaults to 25.
      project_type (str, optional): Type of the instance. Defaults to "blank".
      github_branch (str, optional): Branch of the github repo. Defaults to "".
      github_link (str, optional): Link to the github repo. Defaults to "".
      template_id (int, optional): ID of the template to use. Defaults to 0.
      clone_id (int, optional): ID of the instance to clone. Defaults to 0.

    Returns:
      Instance: The newly created instance.
    """
    workspace_id = workspace_id or secret.workspace_id
    stub_all_projects = nbox_ws_v1.projects

    kwargs_dict = {
      "project_name": project_name,
      "storage_limit": storage_limit,
      "project_type": project_type,
    }
    if github_branch:
      kwargs_dict["github_branch"] = github_branch
    if github_link:
      kwargs_dict["github_link"] = github_link
    if template_id:
      kwargs_dict["template_id"] = template_id
    if clone_id:
      kwargs_dict["clone_id"] = clone_id

    out = stub_all_projects(_method = "post", **kwargs_dict)
    return out

  ################################################################################
  """
  # Utility Methods

  Functions are responsible for doing things that otherwise the user would struggle with.
  """
  ################################################################################

  def get_subway(self, subdomain) -> Sub30:
    """Get a Subway object for the instance.

    Args:
      subdomain (str): The subdomain to connect to
    
    Returns:
      Subway: The Subway object.
    """
    self._unopened_error()

    build = "build"
    if "app.c." in secret.nbx_url:
      build = "build.c"
    elif "app.rc" in secret.nbx_url:
      build = "build.rc"
    url = f"https://{subdomain}-{self.open_data['url']}.{build}.nimblebox.ai/",
    session = Session(
      headers = {
        "NBX-TOKEN": self.open_data["token"],
        "X-NBX-USERNAME": secret.username,
      }
    )
    r = session.get(url + "openapi.json")
    data = r.json()
    sub = Sub30(url, data, session)
    return sub

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
    self.data = self.stub_ws_instance() # GET /user/projects/{instance_id}
    for k in self.useful_keys:
      setattr(self, k, self.data[k])

  def _start(self, cpu, gpu, gpu_count, auto_shutdown, dedicated_hw, zone):
    """Turn on the the unserlying compute"""
    logger.info(f"Starting instance {self.project_name} ({self.instance_id})")
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
      region = zone
    )

    logger.info(f"Waiting for instance {self.project_name} ({self.instance_id}) to start ...")
    _i = 0
    while self.state != "RUNNING":
      time.sleep(5)
      self.refresh()
      _i += 1
      if _i > TIMEOUT_CALLS:
        raise TimeoutError("Instance did not start within timeout, please check dashboard")
    logger.info(f"Instance {self.project_name} ({self.instance_id}) started")

  def _open(self):
    # now the instance is running, we can open it, opening will assign a bunch of cookies and
    # then get us the exact location of the instance
    if not self.__opened:
      logger.debug(f"Opening instance {self.project_name} ({self.instance_id})")
      launch_data = self.stub_ws_instance.launch(_method = "post")
      base_domain = launch_data['base_domain']
      self.open_data = {
        "url": f"{base_domain}",
        "token": self.stub_ws_instance._session.cookies.get_dict()[f"instance_id_{base_domain}"],
        "launch_url": launch_data['launch_url']
      }
      self.__opened = True

  def start(
    self,
    cpu: int = 2,
    gpu: str = "p100",
    gpu_count: int = 0,
    auto_shutdown: int = 6,
    dedicated_hw: bool = False,
    zone = "asia-east1-a",
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
    

    if auto_shutdown < 0:
      raise ValueError("auto_shutdown must be a positive integer (hours)")
    gpu_count = int(gpu_count)

    if not self.state == "RUNNING":
      self._start(cpu, gpu, gpu_count, auto_shutdown, dedicated_hw, zone)
    else:
      # TODO: @yashbonde: inform user in case of hardware mismatch?
      logger.info(f"Instance {self.project_name} ({self.instance_id}) is already running")

    # prevent rate limiting
    if not self.__opened:
      self._open()

  def stop(self):
    """Stop the Instance"""
    if self.state == "STOPPED":
      logger.info(f"Instance {self.project_name} ({self.instance_id}) is already stopped")
      return

    logger.debug(f"Stopping instance {self.project_name} ({self.instance_id})")
    message = self.stub_ws_instance.stop(
      "post",
      data = {"workspace_id": secret.workspace_id, "instance_id": self.instance_id}
    )["msg"]
    if not message == "success":
      raise ValueError(message)

    logger.debug(f"Waiting for instance {self.project_name} ({self.instance_id}) to stop")
    _i = 0 # timeout call counter
    while self.state != "STOPPED":
      time.sleep(5)
      self.refresh()
      _i += 1
      if _i > TIMEOUT_CALLS:
        raise TimeoutError("Instance did not stop within timeout, please check dashboard")
    logger.debug(f"Instance {self.project_name} ({self.instance_id}) stopped")

    self.__opened = False

  def delete(self, force = False):
    """With great power comes great responsibility."""
    if self.__opened and not force:
      raise ValueError("Instance is still opened, please call .stop() first")
    logger.warning(f"Deleting instance {self.project_name} ({self.instance_id})")
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

  def _unopened_error(self):
    if not self.__opened:
      logger.error(f"You are trying to move files to a {self.state} instance, you will have to start the instance first:")
      logger.error(f'    - nbox.Instance("{self.instance_id}", "{self.workspace_id}").start(...)')
      logger.error(f'    - python3 -m nbox build --i "{self.instance_id}" --workspace_id "{self.workspace_id}" start --help')
      raise ValueError("Instance is not opened, please call .open() first")

  def __create_connection(self, *, port: int = 6174):
    """This function is used to create a the Connection Manager object and not depend on `__create_threads` from ssh"""
    from nbox.sub_utils.ssh import FileLogger, ConnectionManager

    # create logging for RSock
    folder = U.join(U.env.NBOX_HOME_DIR(), "tunnel_logs")
    os.makedirs(folder, exist_ok=True)
    filepath = U.join(folder, f"tunnel_{self.instance_id}.log") # consistency with IDs instead of names
    file_logger = FileLogger(filepath)
    logger.debug(f"Logging RSock server to {filepath}")

    # if "app.nimblebox.ai" in self.stub_ws_instance._url:
    #   self.stub_ws_instance._url = self.stub_ws_instance._url.replace("app.", "app.rc.")
    self._open() # open using the updated URL

    conman = ConnectionManager(
      file_logger = file_logger,
      user = secret.username, 
      subdomain = self.open_data.get("url"),
      auth = self.open_data.get("token"),
    )
    conman.add(localport = port, buildport = 2222, _ssh = False)
    return conman

  def __run_command(self, comm: str, port: int, return_output: bool = False) -> str:
    connection = self.__create_connection(port = port)
    try:
      command = shlex.split(comm)
      logger.info(f"Running command: {comm}")

      # there are some commands that require output in which case we run the commands 
      if return_output:
        result = check_output(command, universal_newlines=True)
      else:
        result = run(command, universal_newlines=True)
        result = result.stdout
        result = "" if result is None else result
    except KeyboardInterrupt:
      logger.info("KeyboardInterrupt, closing connections")
      result = ""
    except Exception as e:
      U.log_traceback()
      result = ""

    # Right now we close the connections the moment we are done with the execution, which means
    # that RSock server is started and closed multiple times. Is this a good method, should we
    # open a connection and keep it open till the process is done? Does that also introduce a
    # security risk?
    connection.quit()
    return result.strip()

  def ls(self, path: str, *, port: int = 6174) -> str:
    """
    List files in a directory relative to '/home/ubuntu/project'

    ```
    [nbox API] - nbox.Instance(...).ls("/")
    [nbox CLI] - nbx build ... ls ./
    ```

    Args:
      path (str): Path to list files in all paths will be built relative to "/home/ubuntu/project/" folder.
    """
    self._unopened_error()
    if path.startswith("nbx://"):
      path = path[6:]
    if path == "":
      raise ValueError("Path cannot be empty")
    path = "/home/ubuntu/project/" + path.strip("/")

    # RPC
    logger.info(f"Looking at path: {path}")
    comm = "ssh"
    if U.env.NBOX_SSH_NO_HOST_CHECKING(False):
      comm += " -o StrictHostKeychecking=no"
    comm += f" -p {port} ubuntu@localhost 'ls -l {path}'"
    result = self.__run_command(comm, port, return_output=True)
    return result

  def mv(self, src: str, dst: str, force: bool = False, *, port: int = 6174):
    """
    Move files to and fro NBX-Build. Use 'nbx://' as prefix for Instance, all files will be placed relative to
    '/home/ubuntu/project/' folder.

    Args:
      src (str): Source file or folder to move
      dst (str): Destination file or folder to move to
      force (bool): If True, will override the destination file if it already exists
    """
    self._unopened_error()

    src_is_cloud = src.startswith("nbx://")
    dst_is_cloud = dst.startswith("nbx://")

    if src_is_cloud and dst_is_cloud:
      raise ValueError("Cannot move files between two instances")
    if not src_is_cloud and not dst_is_cloud:
      raise ValueError("Cannot move files on your machine")
    elif src_is_cloud:
      # if the dst is . or ./ it means to copy it to the folder where command is run so that
      # is acceptable but any other path is not acceptable
      if not dst in ["./", "."] and os.path.exists(dst) and not force:
        raise ValueError(f"Source file '{dst}' already exists, pass --force to override")
      ls_res = self.ls(src, port = port)
      src_is_dir = ls_res.startswith("total")
      if not src_is_dir:
        logger.info(f"ls result on cloud: {ls_res}")
    elif dst_is_cloud:
      assert os.path.exists(src), f"Source file '{src}' does not exist"
      # check if this file already exists on the cloud
      ls_res = self.ls(dst, port = port)
      logger.info(f"ls result on cloud: {ls_res}")
      if ls_res:
        logger.info(f"File {dst} already exists")
        if not force:
          logger.error("Aborted, use --force to override")
          return
        logger.info("Overriding file")
      src_is_dir = os.path.isdir(src)

    src = "/home/ubuntu/project/" + src[6:] if src_is_cloud else src
    dst = "/home/ubuntu/project/" + dst[6:] if dst_is_cloud else dst

    # RPC, note order sensitivity for src and dst along with user@host
    logger.info(f"Moving {src} to {dst}")
    comm = "scp"
    if src_is_dir:
      logger.debug(f"Source is a directory")
      comm += " -r" # recursive
    if U.env.NBOX_SSH_NO_HOST_CHECKING(False):
      logger.debug("Host checking is disabled")
      comm += " -o StrictHostKeychecking=no"
    comm += f" -P {port}"
    if src_is_cloud:
      comm += f" ubuntu@localhost:{src} {dst}"
    else:
      comm += f" {src} ubuntu@localhost:{dst}"
    result = self.__run_command(comm, port)
    return result

  def rm(self, file: str, *, port: int = 6174) -> str:
    """
    Remove file from NBX-Build.

    Args:
      file (str): File to remove
    """
    self._unopened_error()

    if not file.startswith("nbx://"):
      raise ValueError("File must be on NBX-Build, try adding 'nbx://' to start!")
    file = "/home/ubuntu/project/" + file.replace("nbx://", "")

    # RPC
    logger.info(f"Removing file: {file}")
    comm = "ssh"
    if U.env.NBOX_SSH_NO_HOST_CHECKING(False):
      comm += " -o StrictHostKeychecking=no"
    comm += f" -p {port} ubuntu@localhost 'rm {file}'"
    result = self.__run_command(comm, port)
    return result

  def remote(self, x: str, *, port: int = 6174) -> str:
    """
    Run any command using SSH, in the underlying system it will setup a SSH connection and execute any command.

    Args:
      x (str): Command to run
    """
    self._unopened_error()

    # RPC
    logger.info(f"Running command: {x}")
    comm = "ssh"
    if U.env.NBOX_SSH_NO_HOST_CHECKING(False):
      comm += " -o StrictHostKeychecking=no"
    comm += f" -p {port} ubuntu@localhost '{x}'"
    result = self.__run_command(comm, port)
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
    # self._unopened_error()
    raise NotImplementedError("Core talking method, not implemented yet")
