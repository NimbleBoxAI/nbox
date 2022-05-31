# If you have come till here, and you want to work on more things like this, reach out:
# research@nimblebox.ai

import os

from nbox.jobs import Job

from ..auth import secret
from ..utils import logger, ENVVARS
from .. import utils as U
from ..init import nbox_grpc_stub
from ..hyperloop.dag_pb2 import Node
from ..hyperloop.job_pb2 import NBXAuthInfo, Job
from ..hyperloop.nbox_ws_pb2 import UpdateRunRequest
from ..messages import rpc

class Tracer:
  def __init__(self):
    run_data = secret.get("run") # user should never have "run" on their local
    if run_data == None:
      self.network_tracer = False
      return
    
    init_folder = ENVVARS.NBOX_JOB_FOLDER(None)
    if init_folder == None:
      raise RuntimeError("NBOX_JOB_FOLDER not set")
    if not os.path.exists(init_folder):
      raise RuntimeError(f"NBOX_JOB_FOLDER {init_folder} does not exist")

    # get this data from the local secrets file
    self.job_id = run_data["job_id"]
    self.token = run_data["token"]
    self.workspace_id = secret.get("workspace_id")
    with open(U.join(init_folder, "job_proto.msg"), "rb") as f:
      self.job_proto = Job()
      self.job_proto.ParseFromString(f.read())

    self.job_proto.id = self.job_id
    self.job_proto.auth_info.CopyFrom(NBXAuthInfo(workspace_id=self.workspace_id))
    self.network_tracer = True
    
    logger.debug(f"Job Id: {self.job_id}")
    logger.debug(f"Run token: {self.token}")
    logger.debug(f"Workspace Id: {self.workspace_id}")
    self.job_proto.status = Job.Status.ACTIVE # automatically first run will

  def __call__(self, node: Node, verbose: bool = True):
    if verbose:
      logger.debug(node)
    if not self.network_tracer:
      return
    self.job_proto.dag.flowchart.nodes[node.id].CopyFrom(node) # even if fails we can keep caching this
    updated_at = node.run_status.start if node.run_status.end != None else node.run_status.end
    logger.info(updated_at.ToDatetime())
    rpc(
      nbox_grpc_stub.UpdateRun,
      UpdateRunRequest(token = self.token, job=self.job_proto, updated_at=updated_at),
      f"Could not update job {self.job_proto.id}"
    )

