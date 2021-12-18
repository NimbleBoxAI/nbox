"""
Jobs
====

Run arbitrary scripts on NimbleBox.ai instances with simplicity of ``nbox``.

This immensely powerful tool will allow methods to be run in a truly serverless fashion.
In the background it will spin up a VM, then run a script and close the machine. This is
the second step towards YoCo and CasH, read more
`here <https://yashbonde.github.io/general-perceivers/remote.html>`_.
"""

import os
import re
import time
from tabulate import tabulate
from requests.sessions import Session

import nbox

# don't use inside instance as cookies can cause a bunch of problems and we'll require
# a session manager
from ..utils import nbox_session
from ..user import secret

from logging import getLogger
logger = getLogger("jobs")

TIMEOUT_CALLS = 60


def get_status(url = "https://nimblebox.ai"):
  r = nbox_session.get(f"{url}/api/instance/get_user_instances")
  r.raise_for_status()
  message = r.json()["msg"]
  if message != "success":
    raise Exception(message)

  money = r.json()["nbBucks"]
  data = [{k: x[k] for k in Instance.useful_keys} for x in r.json()["data"]]
  return money, data

def print_status(url = "https://nimblebox.ai"):
  money, data = get_status(url)
  logger.info(f"Total NimbleBox.ai credits left: {money}")
  data_table = [[x[k] for k in Instance.useful_keys] for x in data]
  for x in tabulate(data_table, headers=Instance.useful_keys).splitlines():
    logger.info(x)

def get_instance(url, id_or_name, session = nbox_session):
  if not isinstance(id_or_name, (int, str)):
    raise ValueError("Instance id must be an integer or a string")

  # now here we can just run .update() method and get away with a lot of code but
  # the idea is that a) during initialisation the user should get full info in case
  # something goes wrong and b) we can use the same code as fallback logic when we
  # want to orchestrate the jobs
  r = session.get(f"{url}/api/instance/get_user_instances")
  r.raise_for_status()
  resp = r.json()

  if len(resp["data"]) == 0:
    raise ValueError("No instance found, create manually from the dashboard or Instance.new(...)")

  key = "instance_id" if isinstance(id_or_name, int) else "name"
  instance = list(filter(lambda x: x[key] == id_or_name, resp["data"]))
  if len(instance) == 0:
    raise KeyError(id_or_name)
  instance = instance[0] # pick the first one

  return instance

def is_random_name(name):
  return re.match(r"[a-z]+-[a-z]+", name) is not None


class Subway():
  def __init__(self, url, session):
    self.url = url
    self.session = session

  def __repr__(self):
    return f"<Subway ({self.url})>"

  def __getattr__(self, attr):
    # https://stackoverflow.com/questions/3278077/difference-between-getattr-vs-getattribute
    def wrapper(method = "get", data = None, params = None, verbose = True):
      fn = getattr(self.session, method)
      url = f"{self.url}/{attr}"
      if verbose:
        logger.info(f"Calling {url}")
      r = fn(url, json = data, params = params)
      if verbose:
        logger.info(r.content.decode()[:100])
      r.raise_for_status() # good when server is good
      return r.json()
    return wrapper

  def __call__(self, end = "", method = "get", data = None, params = None, verbose = True):
    fn = getattr(self.session, method)
    url = f"{self.url}/{end}"
    if verbose:
      logger.info(f"Calling {url}")
    r = fn(url, json = data, params = params)
    if verbose:
      logger.info(r.content.decode()[:100])
    r.raise_for_status() # good when server is good
    return r.json()


# class JobsServerMixin:
#   def files(self, dir_path):

class Instance:
  # each instance has a lot of data against it, we need to store only a few as attributes
  useful_keys = ["state", "used_size", "total_size", "public", "instance_id", "name"]
  
  def __init__(self, id_or_name, loc = None, cs_endpoint = "server"):
    self.url = f"https://{'' if not loc else loc+'.'}nimblebox.ai"
    self.cs_url = None
    self.cs_endpoint = cs_endpoint
    self.session = Session()
    self.session.headers.update({"Authorization": f"Bearer {secret.get('access_token')}"})

    instance = get_instance(self.url, id_or_name, session = self.session)
    # keep only the important ones
    for k,v in instance.items():
      if k in self.useful_keys:
        setattr(self, k, v)
    
    self.data = instance # keep a copy of the whole instance data

    logger.info(f"Instance added: {self.instance_id}, {self.name}")    
    
    self.__opened = False
    self.running_scripts = []

    """There are many APIs in the backend but because we want the user to have a supreme experience
    so we are only keeping the most powerful ones here. However as a developer you'd hate if you
    could not access things that you can otherwise, like uselessly making your life harder. So to
    avoid that bad experience I am adding two classes here (one for webserver and another for Compute
    server), that can be accesed like an attribute but will run an API call.
    """
    self.web_server = Subway(f"{self.url}/api/instance", session = self.session)
    logger.info(f"WS: {self.web_server}")
    if self.state == "RUNNING":
      self.start()

  def __eq__(self, __o: object):
    return self.instance_id == __o.instance_id
  
  def __repr__(self):
    return f"<Instance ({', '.join([f'{k}:{getattr(self, k)}' for k in self.useful_keys + ['cs_url']])})>"

  @classmethod
  def create(cls, name, url = "https://nimblebox.ai"):
    r = nbox_session.post(
      f"{url}/api/instance/create_new_instance_v4",
      json = {"project_name": name, "project_template": "blank"}
    )
    r.raise_for_status() # if its not 200, it's an error
    return cls(name, url)

  def start(self, cpu_only = False, gpu_count = 1):
    if self.__opened:
      logger.info(f"Instance {self.instance_id} is already opened")
      return

    if not self.state == "RUNNING":
      logger.info(f"Starting instance {self.instance_id}")
      message = self.web_server.start_instance(
        "post",
        data = {
          "instance_id": self.instance_id,
          "hw":"cpu" if cpu_only else "gpu",
          "hw_config":{
            "cpu":"n1-standard-8",
            "gpu":"nvidia-tesla-p100",
            "gpuCount": gpu_count,
          }
        }
      )["msg"]
      if not message == "success":
        raise ValueError(message)

      logger.info(f"Waiting for instance {self.instance_id} to start")
      _i = 0 
      while self.state != "RUNNING":
        time.sleep(5)
        self.update()
        _i += 1
        if _i > TIMEOUT_CALLS:
          raise TimeoutError("Instance did not start within timeout, please check dashboard")
      logger.info(f"Instance {self.instance_id} started")
    else:
      logger.info(f"Instance {self.instance_id} is already running")

    # now the instance is running, we can open it, opening will assign a bunch of cookies and
    # then get us the exact location of the instance
    logger.info(f"Opening instance {self.instance_id}")
    instance_url = self.web_server.open_instance(
      "post", data = {"instance_id":self.instance_id}
    )["base_url"].lstrip("/").rstrip("/")
    self.cs_url = f"{self.url}/{instance_url}"
    if self.cs_endpoint:
      self.cs_url += f"/{self.cs_endpoint}"

    # create a subway
    self.compute_server = Subway(self.cs_url, session = self.session)
    logger.info(f"CS: {self.compute_server}")

    # run a simple test and see if everything is working or not
    logger.info(f"Testing instance {self.instance_id}")
    self.compute_server.test()
    self.__opened = True

  def __call__(self, x):
    """Caller is the most important UI/UX. The letter ``x`` in programming is reserved the most
    arbitrary thing, and this ``nbox.Instance`` is the gateway to a cloud instance. You can:
    1. run a script on the cloud
    2. run a local script on the cloud
    3. get the status of a script
    4. run a special kind of functions known as `pure functions <https://en.wikipedia.org/wiki/Pure_function>`_

    Pure functions in programming are functions that are self sufficient in terms of execution,
    eg. all the packages are imported inside the function and there are no side effects in an
    execution (seeds included for probabilistic functions). Writing such functions in python with
    any IDE with checker is dead easy, however the performace guarantee is premium given high
    costs. Thus the logic to parsing these will have to be written as a seperate module.

    Returns:
      [type]: [description]
    """
    if not self.__opened:
      raise ValueError("Instance is not opened, please call .start() first")

    if isinstance(x, str):
      # x as string can mean many different things:
      # 1. uid of an existing script
      # 2. name of file in the local filesystem
      # 3. name of file in the cloud

      if is_random_name(x):
        logger.info(f"Getting status of Job '{x}' on instance {self.instance_id}")
        data = self.compute_server(x, "post")
        if not data["msg"] == "success":
          raise ValueError(data["msg"])
        else:
          status = "RUNNING" if data["status"] else "STOPPED" # /ERRORED
          logger.info(f"Script {x} on instance {self.instance_id} is {status}")
        return status == "RUNNING"
      elif os.path.isfile(x):
        with open(x, "r") as f:
          script = f.read()
        script_name = os.path.split(x)[1]
        logger.info(f"Uploading script {x} ({script_name}) to instance {self.instance_id}")
        data = self.compute_server.start("post", {"script": script, "script_name": script_name})
        if data["msg"] != "success":
          raise ValueError(data["msg"])
        logger.info(f"Script {x} started on instance {self.instance_id} with UID: {data['uid']}")
        self.running_scripts.append(data["uid"])
        return data["uid"]
      else:
        logger.info(f"Trying to find {x} on cloud")
        dir_path, fname = os.path.split(x)
        dir_path = "/mnt/disks/user/project" if not dir_path else dir_path
        fname = os.path.join(dir_path, fname)
        data = self.compute_server.files("post", data = {"dir_path": dir_path})
        if data["msg"] != "success":
          raise ValueError(data["msg"])

        # if the file is not found, or item is directory, then we can't run it
        if fname not in [x[0] for x in data["files"]]:
          raise ValueError(f"File {fname} not found in {dir_path} on cloud")
        if list(filter(lambda x: x[0] == fname, data["files"]))[0][1]:
          raise ValueError(f"File {fname} is a directory on cloud")
        
        logger.info(f"Found {fname} on cloud, executing it")
        data = self.compute_server.start("post", {"script": fname})
        if data["msg"] != "success":
          raise ValueError(data["msg"])
        logger.info(f"Script {x} started on instance {self.instance_id} with UID: {data['uid']}")
        self.running_scripts.append(data["uid"])
        return data["uid"]
    
    elif callable(x):
      # this is the next generation power of nbox, we can pass a function to run on the instance (CasH step1)
      raise ValueError("callable methods are not supported yet, will be included in the next release")
    else:
      raise ValueError("x must be a string or a function")

  def stop(self):
    if self.state == "STOPPED":
      logger.info(f"Instance {self.instance_id} is already stopped")
      return

    logger.info(f"Stopping instance {self.instance_id}")
    message = self.web_server.stop_instance("post", data = {"instance_id":self.instance_id})["msg"]
    if not message == "success":
      raise ValueError(message)

    logger.info(f"Waiting for instance {self.instance_id} to stop")
    _i = 0 # timeout call counter
    while self.state != "STOPPED":
      time.sleep(5)
      self.update()
      _i += 1
      if _i > TIMEOUT_CALLS:
        raise TimeoutError("Instance did not stop within timeout, please check dashboard")
    logger.info(f"Instance {self.instance_id} stopped")

    self.__opened = False

  def delete(self, force = False):
    if self.__opened and not force:
      raise ValueError("Instance is still opened, please call .stop() first")
    logger.info(f"Deleting instance {self.instance_id}")
    message = self.web_server.delete_instance("post", data = {"instance_id":self.instance_id})["msg"]
    if not message == "success":
      raise ValueError(message)

  def update(self):
    out = self.web_server.get_user_instances("post", data = {"instance_id": self.instance_id})
    for k,v in out.items():
      if k in self.useful_keys:
        setattr(self, k, v)
    self.data = out
