# If you have come till here, and you want to work on more things like this, reach out:
# research@nimblebox.ai

import os
import threading
from time import sleep
from json import dumps

import nbox.utils as U
from nbox.utils import logger, SimplerTimes
from nbox.init import nbox_grpc_stub
from nbox.auth import secret, AuthConfig, auth_info_pb
from nbox.hyperloop.jobs.job_pb2 import Job as JobProto
from nbox.hyperloop.jobs.dag_pb2 import Node
from nbox.hyperloop.jobs.nbox_ws_pb2 import UpdateRunRequest, JobRequest
from nbox.messages import rpc, read_file_to_binary, read_file_to_string, message_to_dict

class Tracer:
  def __init__(self, local: bool = False, start_heartbeat: bool = True, heartbeat_every: int = 60):
    self.heartbeat_every = heartbeat_every

    # create the kwargs that are used throughout
    self.job_proto = None
    self.run_id = None
    self.job_id = None
    self.workspace_id = None
    self.network_tracer = False
    self.trace_file = None

    # from the data available in the init folder, get the job proto
    if local:
      pass
    else:
      run_data = secret(AuthConfig.nbx_pod_run) # user should never have "run" on their local
      if run_data is not None:
        self.init(run_data, start_heartbeat)

  @classmethod
  def local(cls, job_proto: JobProto, run_id):
    tracer = Tracer(local = True)
    tracer.job_proto = job_proto
    tracer.run_id = run_id
    tracer.job_id = job_proto.id
    tracer.workspace_id = job_proto.auth_info.workspace_id
    tracer.local_init()
    return tracer

  def local_init(self):
    folder = U.join(f"{U.env.NBOX_HOME_DIR()}", "traces")
    os.makedirs(folder, exist_ok = True)
    file = U.join(folder, f"{self.run_id}.jsonl")

    logger.debug(f"Username: {self.job_proto.auth_info.username}")
    logger.debug(f"Job Id: {self.job_proto.id}")
    logger.debug(f"Run Id: {self.run_id}")
    logger.debug(f"Workspace Id: {self.job_proto.auth_info.workspace_id}")
    self.trace_file = open(file, "a")

  def init(self, run_data, start_heartbeat):
    init_folder = U.env.NBOX_JOB_FOLDER(".")
    if not init_folder:
      raise RuntimeError("NBOX_JOB_FOLDER not set")
    if not os.path.exists(init_folder):
      raise RuntimeError(f"NBOX_JOB_FOLDER {init_folder} does not exist")

    # get this data from the local secrets file
    self.job_id = run_data.get("job_id", None)
    self.run_id = run_data.get("token", None)

    # grandfather old messages (<v0.9.14rc13)
    fp_bin = U.join(init_folder, "job_proto.msg")
    fp_str = U.join(init_folder, "job_proto.pbtxt")
    if os.path.exists(fp_bin):
      self.job_proto: JobProto = read_file_to_binary(fp_bin, JobProto())
    elif os.path.exists(fp_str):
      self.job_proto: JobProto = read_file_to_string(fp_str, JobProto())
    else:
      logger.warning(f"Could not find job_proto.msg or job_proto.pbtxt in {init_folder}, fetching from database")
      self.job_proto = nbox_grpc_stub.GetJob(JobRequest(
        auth_info=auth_info_pb(),
        job = JobProto(id = self.job_id)
      ))

    self.job_proto.id = self.job_id # because when creating a new job, client does not know the ID
    self.workspace_id = secret(AuthConfig._workspace_id)
    self.network_tracer = True

    # logger.debug(f"Username: {self.job_proto.auth_info.username}")
    logger.info(f"Job Id (Run Id) [Workspace ID]: {self.job_id} ({self.run_id}) [{self.workspace_id}]")
    self.job_proto.status = JobProto.Status.ACTIVE # automatically first run will

    # start heartbeat in a different thread
    if start_heartbeat:
      self.thread_stop = threading.Event()
      self.thread = threading.Thread(target=self.hearbeat_thread_worker)
      self.thread.start()

  @property
  def active(self):
    return hasattr(self, "job_proto")

  def __repr__(self) -> str:
    return f"Tracer() for job {self.job_id}"

  def _rpc(self, message: str = ""):
    try:
      rpc(
        nbox_grpc_stub.UpdateRun,
        UpdateRunRequest(
          token = self.run_id,
          job = self.job_proto,
          updated_at = SimplerTimes.get_now_pb(),
          auth_info = auth_info_pb()
        ),
        message or f"Could not update job {self.job_proto.id}",
        raise_on_error = True
      )
    except Exception as e:
      logger.error(f"Could not update job {self.job_proto.id}\n  Error: {e}\n  Most likely could not commumicate with the server")
      U.hard_exit_program(1)

  def __call__(self, node: Node, verbose: bool = False):
    if self.network_tracer:
      self.job_proto.dag.flowchart.nodes[node.id].CopyFrom(node) # even if fails we can keep caching this
      self._rpc()
    else:
      self.trace_file.write(dumps(message_to_dict(node)) + "\n")

  def hearbeat_thread_worker(self):
    while True:
      self._rpc()
      for _ in range(self.hearbeat_thread_worker):
        # in future add a way to stop the thread
        sleep(1)

  def stop(self):
    if not self.network_tracer:
      return
    self.thread.join()
