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

import nbox.utils as U
from nbox import RelicsNBX
from nbox.auth import secret
from nbox.relics.local import RelicLocal
from nbox import Operator, logger
from nbox.operator import Resource
from nbox.utils import log_traceback, SimplerTimes
from nbox.nbxlib.tracer import Tracer
from nbox.hyperloop.job_pb2 import Job
from nbox import Operator, nbox_grpc_stub
from nbox.messages import rpc
from nbox.hyperloop.nbox_ws_pb2 import UpdateRunRequest
from nbox.nbxlib.serving import serve_operator

PROC_FOLDER = "./nbx_autogen_proc"


# Manager
class LocalNBXLet(Operator):
  def __init__(self, op: Operator, in_key: str, out_key: str):
    """This is the Operator which runs on the main Job pod and stores all the values. Think of this like the
    Master node in a distributed system, which contains all the information on the processing."""
    super().__init__()
    self.op = op
    self.in_key = in_key
    self.out_key = out_key

  # def start_status(self):
  #   def worker():
  #     # function to keep on updating the cache and seeing the progress
  #     while True:
  #       available = 0
  #       for key in self.keys:
  #         if self.relic.has(key):
  #           available += 1
  #       if self.verbose:
  #         logger.info(f"Available: [{available}/{len(self.keys)}]")
  #       sleep(0.69)

  #   # start the status thread
  #   self.cache_man = Thread(target = worker, daemon=True)
  #   self.cache_man.start()

  # def prepare_children(self):
  #   self.op_agents = {}
  #   self.keys = set()
  #   for op_name, res in self.op._op_to_resource_map.items():
  #     if res != None:
  #       # replace the Opeartors in the graph with AgentsStubs
  #       op = getattr(self.op, op_name)
  #       agent_op = AgentOpRootStub(op, op_name, res, self.relic)
  #       setattr(self.op, op_name, agent_op)
  #       self.op_agents[op_name] = agent_op
  #       self.keys.add(op_name + "_input")
  #       self.keys.add(op_name + "_output")

  def __repr__(self):
    return f"LocalNBXLet({self.op.__qualname__}, {self.in_key}, {self.out_key})"

  def forward(self):
    x = U.from_pickle(self.in_key)
    y = self.op(*x)
    U.to_pickle(y, self.out_key)

class NBXLet(Operator):
  def __init__(self, op: Operator):
    """The Operator that runs the things on any pod on the NimbleBox Jobs + Deploy platform.
    Name mimics the kubelet, dockerlet, raylet, etc"""
    super().__init__()
    self.op = op

  def run(self):
    """Run this as a batch process"""
    tracer = Tracer()
    secret.put("username", tracer.job_proto.auth_info.username)

    self.op.propagate(_tracer = tracer)
    if hasattr(self.op._tracer, "job_proto"):
      self.op.thaw(self.op._tracer.job_proto)

    workspace_id = tracer.job_proto.auth_info.workspace_id
    job_id = tracer.job_id
    status = Job.Status.ERROR

    try:
      # now for some jobs there might be a relic Object so we can check if that exists, it will always
      # be present in the dot_deploy_cache folder and will be in the {job_id} folder
      relic = RelicsNBX("cache", workspace_id, create = True)

      # check if there is a specific relic for this job
      run_tag = os.getenv("NBOX_RUN_METADATA", None)
      _in = f"{job_id}/args_kwargs"
      if run_tag:
        _in += f"_{run_tag}"
      if relic.has(_in):
        (args, kwargs) = relic.get_object(_in)
      else:
        args, kwargs = (), {}
      out = self.op(*args, **kwargs)
      _out = f"{job_id}/return"
      if run_tag:
        _out += f"_{run_tag}"
      relic.put_object(_out, out)
      status = Job.Status.COMPLETED
    except Exception as e:
      U.log_traceback()
    finally:
      logger.info(f"Job {job_id} completed with status {status}")
      if hasattr(tracer, "job_proto"):
        self.op._tracer.job_proto.status = status
        rpc(
          nbox_grpc_stub.UpdateRun, UpdateRunRequest(
            token = tracer.run_id, job = tracer.job_proto, updated_at = SimplerTimes.get_now_pb()
          ), "Failed to end job!"
        )
      U._exit_program()

  def serve(self, host: str = "0.0.0.0", port: int = 8000, *, model_name: str = None):
    """Run a serving API endpoint"""
    try:
      serve_operator(self.op, host = host, port = port, model_name = model_name)
    except Exception as e:
      U.log_traceback()
      logger.error(f"Failed to serve operator: {e}")
      U._exit_program()
