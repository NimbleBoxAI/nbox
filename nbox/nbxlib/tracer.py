import os
from grpc import RpcError

from nbox.jobs import Job

from ..utils import logger
from ..init import nbox_grpc_stub
from ..hyperloop.dag_pb2 import Node
from ..hyperloop.job_pb2 import NBXAuthInfo, Job
from ..hyperloop.nbox_ws_pb2 import UpdateRunRequest

class Tracer:
  def __init__(self):
    job_id = os.environ.get("NBOX_JOB_ID", False)
    if not job_id:
      self.network_tracer = False
      return
    init_folder = os.getenv("NBOX_JOB_FOLDER", None)
    if init_folder == None:
      raise RuntimeError("NBOX_JOB_FOLDER not set")

    with open(os.path.join(init_folder, "job_proto.msg"), "rb") as f:
      self.job_proto = Job()
      self.job_proto.ParseFromString(f.read())

    self.job_id = self.job_proto.id.upper()
    self.workspace_id = os.getenv("NBOX_WORKSPACE_ID", None)
    self.auth_info = NBXAuthInfo(workspace_id=self.workspace_id)
    self.network_tracer = True
    self.job_proto.status = Job.Status.ACTIVE # automatically first run will

  def __call__(self, node: Node, verbose: bool = True):
    if not self.network_tracer:
      if verbose:
        logger.debug(node)
      return
    self.job_proto.dag.flowchart.nodes[node.id].MergeFrom(node)
    try:
      nbox_grpc_stub.UpdateRun(UpdateRunRequest(job=self.job_proto))
    except RpcError as e:
      logger.error(f"Could not update job {self.id}")
      
