"""
Jobs
====

``nbox.Job`` is a wrapper to the APIs that's it.
"""

import sys
import grpc
import jinja2

from .utils import logger
from . import utils as U
from .init import nbox_grpc_stub
from .auth import secret

import tabulate


from google.protobuf.field_mask_pb2 import FieldMask
from google.protobuf.json_format import MessageToDict

from .hyperloop.job_pb2 import NBXAuthInfo, Job as JobProto
from .hyperloop.nbox_ws_pb2 import JobLog, ListJobsRequest, JobLogsRequest, ListJobsResponse, UpdateJobRequest
from .hyperloop.nbox_ws_pb2 import JobInfo

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
  from .network import Cron

################################################################################
# NBX-Jobs Functions
# ==================
# These functions are assigned as static functions to the ``nbox.Job`` class.
################################################################################

def new(project_name):
  import os, re
  from datetime import datetime
  created_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S") + " UTC"

  out = re.findall("^[a-zA-Z_]+$", project_name)
  if not out:
    raise ValueError("Project name can only contain letters and underscore")

  if os.path.exists(project_name):
    raise ValueError(f"Project {project_name} already exists")

  # ask user requirements here for customisation
  run_on_build = input("> Is this a run on build job? (y/N) ").lower() == "y"

  scheduled = None
  instance = None
  if run_on_build:
    # in this case the job will be run on a nbx-build instance (internal testing)
    # real advantage is ability to run on GPUs and persistent storage
    logger.info("This job will run on NBX-Build")
    instance = input("> Instance name or ID: ").strip()
  else:
    logger.info("This job will run on NBX-Jobs")
    scheduled = input("> Is this a recurring job (y/N)? ").lower() == "y"
    if scheduled:
      logger.info("This job will be scheduled to run on a recurring basis")
    else:
      logger.info(f"This job will run only once")
  
  logger.info(f"Creating a folder: {project_name}")
  os.mkdir(project_name)
  os.chdir(project_name)

  # jinja is cool
  assets = U.join(U.folder(__file__), "assets")
  path = U.join(assets, "job_new.jinja")
  with open(path, "r") as f, open("exe.py", "w") as f2:
    f2.write(
      jinja2.Template(f.read()).render(
        run_on_build = run_on_build,
        project_name = project_name,
        created_time = created_time,
        scheduled = scheduled,
        instance = instance
    ))

  path = U.join(assets, "job_new_readme.jinja")
  with open(path, "r") as f, open("README.md", "w") as f2:
    f2.write(
      jinja2.Template(f.read()).render(
        project_name = project_name,
        created_time = created_time,
        scheduled = scheduled,
    ))

  with open("requirements.txt", "w") as f:
    f.write("nbox==0.8.8a0")

  logger.debug("Completed")

def open_home():
  import webbrowser
  webbrowser.open(secret.get("nbx_url")+"/"+"jobs")

def get_job_list(workspace_id: str = None, filters = "*"):
  auth_info = NBXAuthInfo(workspace_id = workspace_id)
  try:
    out: ListJobsResponse = nbox_grpc_stub.ListJobs(ListJobsRequest(auth_info = auth_info))
  except grpc.RpcError as e:
    logger.error(f"{e.details()}")
    sys.exit(1)

  out = MessageToDict(out)
  if len(out) == 0:
    logger.info("No jobs found")
  
  headers=list(out["Jobs"][0].keys())
  data = []
  for j in out["Jobs"]:
    if filters == "*":
      data.append([j[x] for x in headers])
    if j["status"] in filters:
      data.append([j[x] for x in headers])
  for l in tabulate.tabulate(data, headers).splitlines():
    logger.info(l)



################################################################################
# NimbleBox.ai Jobs
# =================
# This is the actual job object that users can manipulate. It is a shallow class
# around the NBX-Jobs gRPC API.
################################################################################

class Job:
  def __init__(self, id, workspace_id = None):
    """Python wrapper for NBX-Jobs gRPC API

    Args:
        id (str): job ID
        workspace_id (str, optional): If None personal workspace is used. Defaults to None.
    """
    self.id = id
    self.workspace_id = workspace_id
    auth_info = NBXAuthInfo(workspace_id = workspace_id)
    self.job_proto = JobProto(id = id, auth_info = auth_info)
    self.update()

  # static methods
  new = staticmethod(new)
  home = staticmethod(open_home)
  status = staticmethod(get_job_list)

  def change_schedule(self, new_schedule: 'Cron'):
    # nbox should only request and server should check if possible or not
    pass

  def __repr__(self) -> str:
    return f"nbox.Job('{self.job_proto.id}', '{self.job_proto.auth_info.workspace_id}')"

  def stream_logs(self, f = sys.stdout):
    # this function will stream the logs of the job in anything that can be written to
    logger.info(f"Streaming logs of job {self.job_proto.id}")
    try:
      log_iter: List[JobLog] = nbox_grpc_stub.GetJobLogs(JobLogsRequest(job = JobInfo(job = self.job_proto)))
      for job_log in log_iter:
        for log in job_log.log:
          f.write(log + "\n")
          f.flush()
    except grpc.RpcError as e:
      logger.error(f"Could not get logs of job {self.job_proto.id}, is your job complete?")
      logger.error(e.details())

  def delete(self):
    logger.info(f"Deleting job {self.job_proto.id}")
    try:
      nbox_grpc_stub.DeleteJob(JobInfo(job = self.job_proto,))
    except grpc.RpcError as e:
      logger.error(f"Could not delete job {self.job_proto.id}")
      logger.error(e.details())
      raise e

  def update(self):
    logger.info("Updating job info")
    try:
      job: JobProto = nbox_grpc_stub.GetJob(JobInfo(job = self.job_proto))
    except grpc.RpcError as e:
      logger.error(f"Could not get job {self.job_proto.id}")
      logger.error(e.details())
      raise e
    for descriptor, value in job.ListFields():
      setattr(self, descriptor.name, value)
    self.job_proto = job
    self.job_proto.auth_info.CopyFrom(NBXAuthInfo(workspace_id = self.workspace_id))

  def trigger(self):
    logger.info(f"Triggering job {self.job_proto.id}")
    try:
      nbox_grpc_stub.TriggerJob(JobInfo(job=self.job_proto))
    except grpc.RpcError as e:
      logger.error(f"Could not trigger job {self.job_proto.id}")
      logger.error(e.details())
      raise e
  
  def pause(self):    
    logger.info(f"Pausing job {self.job_proto.id}")
    try:
      job: JobProto = self.job_proto
      job.status = JobProto.Status.PAUSED
      job.paused = True
      update_mask = FieldMask(paths=["status", "paused"])
      nbox_grpc_stub.UpdateJob(UpdateJobRequest(job=job, update_mask=update_mask))
    except grpc.RpcError as e:
      logger.error(f"Could not pause job {self.job_proto.id}")
      logger.error(e.details())
      raise e
    logger.info(f"Paused job {self.job_proto.id}")
  
  def resume(self):
    logger.info(f"Resuming job {self.job_proto.id}")
    try:
      job: JobProto = self.job_proto
      job.status = JobProto.Status.SCHEDULED
      job.paused = False
      update_mask = FieldMask(paths=["status", "paused"])
      nbox_grpc_stub.UpdateJob(UpdateJobRequest(job=job, update_mask=update_mask))
    except grpc.RpcError as e:
      logger.error(f"Could not resume job {self.job_proto.id}")
      logger.error(e.details())
      raise e
    logger.info(f"Resumed job {self.job_proto.id}")
