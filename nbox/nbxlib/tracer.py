# If you have come till here, and you want to work on more things like this, reach out:
# research@nimblebox.ai

import os
from time import sleep

import threading

from nbox.jobs import Job

from .. import utils as U
from ..utils import logger
from ..init import nbox_grpc_stub
from ..auth import secret
from ..hyperloop.job_pb2 import Job
from ..hyperloop.dag_pb2 import Node
from ..hyperloop.nbox_ws_pb2 import UpdateRunRequest
from ..messages import rpc, get_current_timestamp, read_file_to_binary, read_file_to_string

class Tracer:
  def __init__(self, heartbeat_every: int = 60):
    self.heartbeat_every = heartbeat_every
    run_data = secret.get("run") # user should never have "run" on their local
    if run_data == None:
      self.network_tracer = False
      return
    
    init_folder = U.ENVVARS.NBOX_JOB_FOLDER(None)
    if init_folder == None:
      raise RuntimeError("NBOX_JOB_FOLDER not set")
    if not os.path.exists(init_folder):
      raise RuntimeError(f"NBOX_JOB_FOLDER {init_folder} does not exist")

    # get this data from the local secrets file
    self.job_id = run_data["job_id"]
    self.token = run_data["token"]
    
    # grandfather old messages (<v0.9.14rc13)
    fp_bin = U.join(init_folder, "job_proto.msg")
    fp_str = U.join(init_folder, "job_proto.pbtxt")
    if os.path.exists(fp_bin):
      self.job_proto = read_file_to_binary(fp_bin, Job())
    elif os.path.exists(fp_str):
      self.job_proto = read_file_to_string(fp_str, Job())
    else:
      raise RuntimeError("Could not find job_proto.msg or job_proto.pbtxt")

    self.job_proto.id = self.job_id # because when creating a new job, client does not know the ID
    self.workspace_id = self.job_proto.auth_info.workspace_id
    self.network_tracer = True
    
    logger.debug(f"Username: {self.job_proto.auth_info.username}")
    logger.debug(f"Job Id: {self.job_proto.id}")
    logger.debug(f"Run Id: {self.token}")
    logger.debug(f"Workspace Id: {self.job_proto.auth_info.workspace_id}")
    self.job_proto.status = Job.Status.ACTIVE # automatically first run will

    # start heartbeat in a different thread
    self.thread = threading.Thread(target=self.hearbeat_thread_worker)
    self.thread.start()

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
      UpdateRunRequest(token = self.token, job = self.job_proto, updated_at = updated_at),
      f"Could not update job {self.job_proto.id}",
      raise_on_error = False
    )

  def hearbeat_thread_worker(self):
    while True:
      rpc(
        nbox_grpc_stub.UpdateRun,
        UpdateRunRequest(token = self.token, job = self.job_proto, updated_at = get_current_timestamp()),
        "Heartbeat failed",
        raise_on_error = False
      )
      sleep(self.heartbeat_every)

  def stop(self):
    if not self.network_tracer:
      return
    self.thread.join()
