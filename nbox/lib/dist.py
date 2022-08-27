"""
With NimbleBox you can run cluster wide workloads from anywhere. This requires capabilities around distributed computing,
process managements. The code here is tested along with ``nbox.Relic`` to perform distributed local processing.
"""

import os
import inspect
import subprocess
from time import sleep

from threading import Thread
from typing import Union

from nbox import RelicsNBX
from nbox.relics.local import RelicLocal
from nbox import Operator, logger
from nbox.operator import Resource
from nbox.utils import log_traceback

PROC_FOLDER = "./nbx_autogen_proc"


# Manager
class RootRunner(Operator):
  def __init__(
    self,
    op: Operator,
    dist = False,
    relic_type = "nbx",
    verbose: str = False,
    *,
    _unittest: bool = False
  ):
    """This is the Operator which runs on the main Job pod and stores all the values. Think of this like the
    Master node in a distributed system, which contains all the information on the processing."""
    super().__init__()
    self.op = op
    self.dist = dist
    self.verbose = verbose
    self._unittest = _unittest

    _relic_cls = {"local": RelicLocal, "nbx": RelicsNBX}.get(relic_type, None)
    if _relic_cls is None:
      raise ValueError(f"Cannot find relic type: '{relic_type}'")

    self.relic = _relic_cls(relic_name = "hustle_bustle", workspace_id = "wnja9glc", create = True)
    self.cache_status = {}

  def start_status(self):
    def worker():
      # function to keep on updating the cache and seeing the progress
      while True:
        available = 0
        for key in self.keys:
          if self.relic.has(key):
            available += 1
        if self.verbose:
          logger.info(f"Available: [{available}/{len(self.keys)}]")
        sleep(0.69)

    # start the status thread
    self.cache_man = Thread(target = worker, daemon=True)
    self.cache_man.start()

  def prepare_children(self):
    self.op_agents = {}
    self.keys = set()
    for op_name, res in self.op._op_to_resource_map.items():
      if res != None:
        # replace the Opeartors in the graph with AgentsStubs
        op = getattr(self.op, op_name)
        agent_op = AgentOpRootStub(op, op_name, res, self.relic)
        setattr(self.op, op_name, agent_op)
        self.op_agents[op_name] = agent_op
        self.keys.add(op_name + "_input")
        self.keys.add(op_name + "_output")

  def clear_relic(self):
    for op_name in self.op_agents:
      for _x in ["_input", "_output"]:
        key = op_name + _x
        if self.relic.has(key):
          self.relic.rm(key)

  def forward(self, *args, **kwargs):
    if self.dist:
      self.prepare_children()
      if self._unittest:
        self.clear_relic()
      self.start_status()

    return self.op(*args, **kwargs)


class AgentOpRootStub(Operator):
  def __init__(self, op: Operator, op_name:str , res: Resource, relic: Union[RelicLocal, RelicsNBX]):
    """This is the root stub for the remote job. In the current implementation it will spin up a new process
    and manage the connections and things for it.
    
    Hmmm... One day maybe [run go in python](https://medium.com/analytics-vidhya/running-go-code-from-python-a65b3ae34a2d)
    
    Think of this like kubelet or raylet"""
    super().__init__()
    self.op = op
    self.op_name = op_name
    self.relic = relic
    logger.info(f"Initialising AgentOpRootStub: {self.op_name}")

  def get_keys(self):
    input_key = self.op_name +  "_input"
    output_key = self.op_name + "_output"
    return input_key, output_key

  def forward(self, *args, **kwargs):
    # take this op and deploy this to a remote machine where it will start
    # and wait for the remote to finish

    # print(self.relic._objects)
    input_key, output_key = self.get_keys()
    if self.relic.has(output_key):
      self.relic.rm(output_key)
    logger.info(f"{input_key} / {output_key}")

    if not os.path.exists(PROC_FOLDER):
      os.mkdir(PROC_FOLDER)

    fpath = os.path.join(PROC_FOLDER, "auto_" + self.op_name + ".py")
    logger.info(f"Creating file: {fpath}")
    with open(fpath, "w") as f:
      f.write(f'''# Auto generated
import os

os.environ["NBOX_LOG_LEVEL"] = "debug"

from nbox import operator
from nbox.relics import RelicLocal, RelicsNBX
from nbox.utils import load_module_from_path

@operator()
def run():
  # load the operator
  op_cls = load_module_from_path("{self.op.__class__.__qualname__}", "{inspect.getfile(self.op.__class__)}")
  op = op_cls()

  # define the keys
  op_name = "{self.op_name}"
  input_key = "{input_key}"
  output_key = "{output_key}"
  
  # define relic and get things
  relic = RelicsNBX(relic_name = "hustle_bustle", workspace_id = "wnja9glc", create = True)
  args, kwargs = relic.get_object(input_key)
  
  out = op(*args, **kwargs)
  if out == None:
    out = "NBX_NULL"

  # print("Putting", output_key, out)
  relic.put_object(output_key, out)

if __name__ == "__main__":
  # if required user can update the status of the operator
  # run.thaw()
  run()
''')

    # put the data in the object store
    self.relic.put_object(input_key, (list(args), kwargs))

    # run the file as a subprocess: pseudo parallel
    try:
      out = subprocess.run(["python3", fpath], universal_newlines=True)
      if out.returncode != 0:
        raise Exception(f"Process returned non zero code: {out.returncode}")
    except KeyboardInterrupt:
      logger.info("KeyboardInterrupt, closing connections")
    except Exception as e:
      logger.error(f"Error in process: '{self.op_name}', check above for details")
      log_traceback()
      raise e

    # get the results from the object store
    out = self.relic.get_object(output_key)
    while out == None:
      sleep(1)
      out = self.relic.get_object(output_key)
    return out

