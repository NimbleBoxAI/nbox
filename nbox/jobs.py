"""
Jobs
====

Run arbitrary scripts on NimbleBox.ai instances with simplicity of ``nbox``.

This immensely powerful tool will allow methods to be run in a truly serverless fashion.
In the background it will spin up a VM, then run a script and close the machine. This is
the second step towards YoCo, read more
`here <https://yashbonde.github.io/general-perceivers/remote.html>`_.
"""

import time

from .utils import nbox_session

from logging import getLogger
logger = getLogger(__name__)

TIMEOUT_CALLS = 60

class Instance:
  useful_keys = ["state", "used_size", "total_size", "public", "instance_id", "name"]
  def __init__(self, id_or_name, url = "https://nimblebox.ai"):
    self.url = url
    self.cs_url = None
    
    # now here we can just run .update() method and get away with a lot of code but
    # the idea is that a) during initialisation the user should get full info in case
    # something goes wrong and b) we can use the same code fallback logic when we want
    # to orchestrate the jobs
    r = nbox_session.get(f"{self.url}/api/instance/get_user_instances")
    r.raise_for_status()
    resp = r.json()

    if len(resp["data"]) == 0:
      raise ValueError("No instance found, please create one manually one from the dashboard")

    if id_or_name is not None:
      if isinstance(id_or_name, int):
        instance = list(filter(lambda x: x["instance_id"] == id_or_name, resp["data"]))
      elif isinstance(id_or_name, str):
        instance = list(filter(lambda x: x["name"] == id_or_name, resp["data"]))
      else:
        raise TypeError("id_or_name must be int or str")
      if len(instance) == 0:
        raise KeyError(f"No instance found with id_or_name: '{id_or_name}'")
      instance = instance[0] # pick the first one

    for k,v in instance.items():
      if k in self.useful_keys:
        setattr(self, k, v)

    if self.state == "RUNNING":
      self.open()

    logger.info(f"Instance added: {self.instance_id}, {self.name}")

  def __eq__(self, __o: object):
    return self.instance_id == __o.instance_id
  
  def __repr__(self):
    return f"<Instance ({', '.join([f'{k}:{getattr(self, k)}' for k in self.useful_keys + ['cs_url']])})>"

  def start(self, cpu_only = False, gpu_count = 1, wait = False):
    if self.state == "RUNNING":
      logger.info(f"Instance {self.instance_id} is already running")
      self.open()
      return

    logger.info(f"Starting instance {self.instance_id}")
    r = nbox_session.post(
      f"{self.url}/api/instance/start_instance",
      json = {
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

    if wait:
      _i = 0 # timeout call counter
      logger.info(f"Waiting for instance {self.instance_id} to start")
      while self.state != "RUNNING":
        time.sleep(5)
        self.update()
        _i += 1
        if _i > TIMEOUT_CALLS:
          raise TimeoutError("Instance did not start within timeout, please check dashboard")
      logger.info(f"Instance {self.instance_id} started")

  def stop(self, wait = False):
    if self.state == "STOPPED":
      logger.info(f"Instance {self.instance_id} is already stopped")
      return

    logger.info(f"Stopping instance {self.instance_id}")
    r = nbox_session.post(
      f"{self.url}/api/instance/stop_instance",
      json = {"instance_id":self.instance_id,}
    )
    message = r.json()["msg"]
    if not message == "success":
      raise ValueError(message)

    if wait:
      _i = 0 # timeout call counter
      logger.info(f"Waiting for instance {self.instance_id} to stop")
      while self.state != "STOPPED":
        time.sleep(5)
        self.update()
        _i += 1
        if _i > TIMEOUT_CALLS:
          raise TimeoutError("Instance did not stop within timeout, please check dashboard")
      logger.info(f"Instance {self.instance_id} stopped")

  def open(self):
    logger.info(f"Opening instance {self.instance_id}")
    r = nbox_session.post(
      f"{self.url}/api/instance/open_instance",
      json = {"instance_id":self.instance_id,}
    )
    instance_url = r.json()["base_url"].lstrip("/").rstrip("/")
    self.cs_url = f"{self.url}/{instance_url}/server"
    r = nbox_session.get(f"{self.cs_url}/test")
    r.raise_for_status()

  def get_files(self, dir = "/mnt/disks/user/project"):
    if self.cs_url == None:
      self.open()

    r = nbox_session.post(
      f"{self.cs_url}/get_files",
      json = {"dir_path": dir,}
    )
    print(r.content)

  def run_script(self, script_path):
    if self.cs_url == None:
      self.open()

    logger.info(f"Running script {script_path} on instance {self.instance_id}")
    r = nbox_session.post(
      f"{self.cs_url}/run_script",
      json = {"script": script_path,}
    )
    print(r.content)
    message = r.json()["msg"]
    if not message == "success":
      raise ValueError(message)

  def test(self):
    if self.cs_url == None:
      self.open()

    r = nbox_session.get(f"{self.cs_url}/test")
    print("ASDFASDFASDFDASF", r.content)
    r.raise_for_status()

  def get_script_status(self, script_path):
    if self.cs_url == None:
      self.open()

    logger.info(f"Getting status of script {script_path} on instance {self.instance_id}")
    r = nbox_session.post(
      f"{self.cs_url}/get_script_status",
      json = {"script": script_path,}
    )
    print(r.content)
    message = r.json()["msg"]
    if not message == "success":
      raise ValueError(message)

  def update(self):
    r = nbox_session.post(
      f"{self.url}/api/instance/get_user_instances",
      json = {
        "instance_id": self.instance_id,
      }
    )
    print(r.content)
    r.raise_for_status()
    for k,v in r.json().items():
      if k in self.useful_keys:
        setattr(self, k, v)
