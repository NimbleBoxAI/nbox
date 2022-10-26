"""
With NimbleBox you can run cluster wide workloads from anywhere. This requires capabilities around distributed computing,
process managements. The code here is tested along with ``nbox.Relic`` to perform distributed local processing.
"""

import os

import nbox.utils as U
from nbox import RelicsNBX
from nbox.auth import secret, ConfigString
from nbox import Operator, logger
from nbox.utils import SimplerTimes
from nbox.nbxlib.tracer import Tracer
from nbox.hyperloop.job_pb2 import Job
from nbox import Operator, nbox_grpc_stub
from nbox.messages import rpc
from nbox.hyperloop.nbox_ws_pb2 import UpdateRunRequest
from nbox.nbxlib.serving import serve_operator


# Manager
class LocalNBXLet(Operator):
  def __init__(self, op: Operator, in_key: str, out_key: str):
    super().__init__()
    self.op = op
    self.in_key = in_key
    self.out_key = out_key

  def __repr__(self):
    return f"LocalNBXLet({self.op.__qualname__}, {self.in_key}, {self.out_key})"

  def forward(self):
    x = U.from_pickle(self.in_key)
    y = self.op(*x)
    U.to_pickle(y, self.out_key)


class NBXLet(Operator):
  def __init__(self, op: Operator):
    """The Operator that runs the things on any pod on the NimbleBox Jobs + Deploy platform.
    Name a parody of kubelet, dockerlet, raylet, etc"""
    super().__init__()
    self.op = op

  def run(self):
    """Run this as a batch process"""
    status = Job.Status.ERROR
    try:
      tracer = Tracer()
      if hasattr(self.op._tracer, "job_proto"):
        self.op.thaw(self.op._tracer.job_proto)
      workspace_id = tracer.job_proto.auth_info.workspace_id
      secret.put(ConfigString.workspace_id, workspace_id, True)
      secret.put("username", tracer.job_proto.auth_info.username)

      job_id = tracer.job_id
      self.op.propagate(_tracer = tracer)

      # get the user defined tag 
      run_tag = os.getenv("NBOX_RUN_METADATA", "")
      logger.info(f"Tag: {run_tag}")

      # in the NimbleBox system we provide tags for each key which essentially tells what is the behaviour
      # of the job. For example if it contains the string LMAO which means we need to initialise a couple
      # of things, or this can be any other job type
      from nbox.lmao import LMAO_JOB_TYPE_PREFIX, LMAO_RELIC_NAME, _lmaoConfig
      if run_tag.startswith(LMAO_JOB_TYPE_PREFIX):
        relic = RelicsNBX(LMAO_RELIC_NAME, workspace_id)
        fp = run_tag[len(LMAO_JOB_TYPE_PREFIX)+1:] # +1 for the -
        if not relic.has(fp+"/init.pkl"):
          raise Exception(f"Could not find init.pkl for tag {run_tag}")
        init_data = relic.get_object(fp+"/init.pkl")
        _lmaoConfig.kv = init_data
        args = _lmaoConfig.kv["args"]
        kwargs = _lmaoConfig.kv["kwargs"]
        # envs = _lmaoConfig.kv["envs"]
        logger.info(_lmaoConfig.kv)
      else:
        # check if there is a specific relic for this job
        relic = RelicsNBX("cache", workspace_id)
        _in = f"{job_id}/args_kwargs"
        if run_tag:
          _in += f"_{run_tag}"
        if relic.has(_in):
          (args, kwargs) = relic.get_object(_in)
        else:
          args, kwargs = (), {}

      # call the damn thing
      out = self.op(*args, **kwargs)

      # save the output to the relevant place
      if run_tag.startswith(LMAO_JOB_TYPE_PREFIX):
        _out = fp+"/return.pkl"
      else:
        _out = f"{job_id}/return"
        if run_tag:
          _out += f"_{run_tag}"
      relic.put_object(_out, out)

      # last step mark as completed
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
