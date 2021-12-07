"""
Jobs
====

Run arbitrary scripts on NimbleBox.ai instances with simplicity of ``nbox``.

This immensely powerful tool will allow methods to be run in a truly serverless fashion.
In the background it will spin up a VM, then run a script and close the machine. This is
the second step towards YoCo and CasH, read more
`here <https://yashbonde.github.io/general-perceivers/remote.html>`_.
"""

import time

from .utils import nbox_session

from logging import getLogger
logger = getLogger(__name__)

TIMEOUT_CALLS = 60
    
class Subway():
  def __init__(self, url):
    self.url = url

  def __repr__(self):
    return f"<Subway ({self.url})>"

  def __getattr__(self, attr):
    def wrapper(method = "get", data = None, verbose = False):
      fn = getattr(nbox_session, method)
      url = f"{self.url}/{attr}"
      if verbose:
        logger.info(f"Calling {url}")
      return fn(url, json = data)
    return wrapper


class Instance:
  # each instance has a lot of data against it, we need to store only a few as attributes
  useful_keys = ["state", "used_size", "total_size", "public", "instance_id", "name"]
  
  def __init__(self, id_or_name, url = "https://nimblebox.ai"):
    self.url = url
    self.cs_url = None
    
    # now here we can just run .update() method and get away with a lot of code but
    # the idea is that a) during initialisation the user should get full info in case
    # something goes wrong and b) we can use the same code as fallback logic when we
    # want to orchestrate the jobs
    r = nbox_session.get(f"{self.url}/api/instance/get_user_instances")
    r.raise_for_status()
    resp = r.json()

    if len(resp["data"]) == 0:
      raise ValueError("No instance found, create manually from the dashboard or Instance.new(...)")

    if id_or_name is not None:
      if not isinstance(id_or_name, (int, str)):
        raise ValueError("Instance id must be an integer or a string")
      key = "instance_id" if isinstance(id_or_name, int) else "name"
      instance = list(filter(lambda x: x[key] == id_or_name, resp["data"]))
      if len(instance) == 0:
        raise KeyError(f"No instance found with id_or_name: '{id_or_name}'")
      instance = instance[0] # pick the first one

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
    self.web_server = Subway(f"{url}/api/instance")
    logger.info(f"WS: {self.web_server}")
    if self.state == "RUNNING":
      self.start()


  def __eq__(self, __o: object):
    return self.instance_id == __o.instance_id
  
  def __repr__(self):
    return f"<Instance ({', '.join([f'{k}:{getattr(self, k)}' for k in self.useful_keys + ['cs_url']])})>"

  """There are many actions that the user can take for each instance, here is a list of all of them
  1. create: /create_new_instance_v4
  2. start: /start_instance
  3. open: /open_instance
  6. stop: /stop_instance
  7. delete: /delete_instance
  
  4. run script
  5. get script status

  Now it is easy to do all this from the frontend, but it feels very clumsy when you have to run
  this thing from code. The entire purpose of nbox is to make ML chill, we cannot accept user making
  so many steps and writing so much code. So I have reudced the steps by merging them into the
  following:

  1.  i = Instance.new(...): When user wants to create a new instance
  1A. i = Instance(...): When loading an existing instance
  2.  i.start(...): To start the instance with some configuration -> will wait for starting then open
  3.  i(...): __call__ override ensures the simple syntax, user can pass a function or script
  4.  i.stop(...): To stop the instance
  4A. i.delete(...): To delete the instance -> i.stop() then delete
  """

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
      r = self.web_server.start_instance(
        method = "post",
        data = {
          "instance_id": self.instance_id,
          "hw":"cpu" if cpu_only else "gpu",
          "hw_config":{
            "cpu":"n1-standard-8",
            "gpu":"nvidia-tesla-p100",
            "gpuCount": gpu_count,
          }
        }
      )
      message = r.json()["msg"]
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
    r = self.web_server.open_instance(method = "post", data = {"instance_id":self.instance_id})
    r.raise_for_status()
    instance_url = r.json()["base_url"].lstrip("/").rstrip("/")
    self.cs_url = f"{self.url}/{instance_url}/server"

    # create a subway
    self.compute_server = Subway(self.cs_url)
    logger.info(f"CS: {self.compute_server}")

    # run a simple test and see if everything is working or not
    logger.info(f"Testing instance {self.instance_id}")
    r = self.compute_server.test()
    r.raise_for_status()

    self.__opened = True

  def __call__(self, path_or_func):
    if not self.__opened:
      raise ValueError("Instance is not opened, please call .start() first")

    if isinstance(path_or_func, str):
      if path_or_func not in self.running_scripts:
        # this script is not running, so we will start it
        logger.info(f"Running script {path_or_func} on instance {self.instance_id}")
        r = self.compute_server.run_script(method = "post", data = {"script_path": path_or_func})
        r.raise_for_status()
        message = r.json()["msg"]
        if not message == "success":
          raise ValueError(message)
        self.running_scripts.append(path_or_func)
      else:
        # we already have this script running, so get the status of this script
        logger.info(f"Getting status of script {path_or_func} on instance {self.instance_id}")
        r = self.compute_server.get_script_status(method = "post", data = {"script": path_or_func})
        message = r.json()["msg"]
        if not message == "script running":
          raise ValueError(message)
        elif "script not running" in message:
          logger.info(f"Script {path_or_func} on instance {self.instance_id} is either completed or errored out.")
        else:
          raise ValueError(message)

    elif callable(path_or_func):
      # this is the next generation power of nbox, we can pass a function to run on the instance (CasH step1)
      raise ValueError("callable methods are not supported yet, will be included in the next release")
    else:
      raise ValueError("path_or_func must be a string or a function")

  def stop(self):
    if self.state == "STOPPED":
      logger.info(f"Instance {self.instance_id} is already stopped")
      return

    logger.info(f"Stopping instance {self.instance_id}")
    r = self.web_server.stop_instance(method="post", data = {"instance_id":self.instance_id})
    message = r.json()["msg"]
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
    
    r = self.web_server.delete_instance(method = "post", data = {"instance_id":self.instance_id})
    r.raise_for_status()
    message = r.json()["msg"]
    if not message == "success":
      raise ValueError(message)

  def update(self):
    r = self.web_server.get_user_instances(method = "post", data = {"instance_id": self.instance_id})
    r.raise_for_status()
    for k,v in r.json().items():
      if k in self.useful_keys:
        setattr(self, k, v)
    self.data = r.json()
