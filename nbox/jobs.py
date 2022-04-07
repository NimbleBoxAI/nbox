"""
``nbox.Job`` is a wrapper to the APIs that's it.
"""

import re
import sys
import os, re
import jinja2
import tabulate
from datetime import datetime, timezone
from google.protobuf.field_mask_pb2 import FieldMask

from . import utils as U
from .utils import logger
from .instance import Instance
from .version import __version__
from .init import nbox_grpc_stub
from .hyperloop.nbox_ws_pb2 import JobInfo
from .hyperloop.job_pb2 import NBXAuthInfo, Job as JobProto
from .hyperloop.nbox_ws_pb2 import ListJobsRequest, JobLogsRequest, ListJobsResponse, UpdateJobRequest
from .messages import message_to_dict, rpc, streaming_rpc


################################################################################
# NBX-Jobs Functions
# ==================
# These functions are assigned as static functions to the ``nbox.Job`` class.
################################################################################

def _nbx_job(project_name: str, workspace_id: str = None):
  from .network import Schedule
  # https://blog.ganssle.io/articles/2019/11/utcnow.html
  created_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S") + " UTC"

  job_id_or_name = input("> Job ID or name: ")

  scheduled = None
  logger.info("This job will run on NBX-Jobs")
  scheduled = input("> Is this a recurring job (y/N)? ").lower() == "y"
  cron_instr = None
  if scheduled:
    logger.info("Calendar Instructions (all time is in UTC Timezone):")
    logger.info("\"   every 4:30 at friday\": Schedule(4, 30, ['fri'])")
    logger.info("\"every 12:00 on weekends\": Schedule(12, 0, ['sat', 'sun'])")
    logger.info("\"         every 10 hours\": Schedule(10)")
    logger.info("\"      every 420 minutes\": Schedule(minute = 420)")
    logger.info("> Enter the calender instruction: ")
    cron_instr = input("> ").strip()
    try:
      eval(cron_instr, {'Schedule': Schedule})
    except Exception as e:
      logger.error(f"Invalid calender instruction: {e}")
      sys.exit(1)

  logger.info("This job will be scheduled to run on a recurring basis" if scheduled else "This job will run once")
  logger.info(f"Creating a folder: {project_name}")
  os.mkdir(project_name)
  os.chdir(project_name)
  py_data = dict(
    import_string_nbox = "from nbox.network import Schedule" if scheduled else None,
    job_id_or_name = job_id_or_name,
    workspace_id = workspace_id,
    scheduled = cron_instr,
    project_name = project_name,
    created_time = created_time,
  )
  py_f_data = {k:v for k,v in py_data.items() if v is not None}

  assets = U.join(U.folder(__file__), "assets")
  path = U.join(assets, "job_new.jinja")
  with open(path, "r") as f, open("exe.py", "w") as f2:
    f2.write(jinja2.Template(f.read()).render(**py_f_data))

  md_data = dict(
    project_name = project_name,
    created_time = created_time,
    scheduled = scheduled,
  )

  path = U.join(assets, "job_new_readme.jinja")
  with open(path, "r") as f, open("README.md", "w") as f2:
    f2.write(jinja2.Template(f.read()).render(**md_data))

def _build_job(project_name, workspace_id):
  created_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S") + " UTC"
  project_id = input("> Project ID: ")
  inst = Instance(i = project_id, workspace_id = workspace_id)
  cpu, gpu_name, gpu_count = None, None, None
  if not inst.status == "RUNNING":
    logger.info("H/W Config Options:")
    logger.info("Since the instance is not running, you can select the hardware config for this job.")
    logger.info("there are two options (if no gpu is provided runs on cpu only):")
    logger.info("--cpu=2 [--gpu='<gpu_name>:<gpu_count>']")
    hw_config = input("> ").strip()
    splits = hw_config.split()
    if len(hw_config) < 3:
      logger.error(f"Invalid hardware config: {hw_config}")
      sys.exit(1)

    cpu = splits[0]
    gpu = None if len(splits) == 1 else splits[1]
    try:
      cpu = re.findall("^--cpu=(\d+)$", cpu)
      cpu = int(cpu)
      if gpu:
        gpu_name, gpu_count = re.findall("^--gpu='(.+):(\d+)'$", gpu)
        gpu_count = int(gpu_count)
    except:
      logger.error(f"Invalid hardware config: {hw_config}")
      sys.exit(1)
  
  py_data = dict(
    run_on_build = True,
    instance = None,
    import_string_others = "import subprocess",
    import_string_nbox = "from nbox.jobs import Instance",
    not_running = inst.status != "RUNNING",
    cpu_only = cpu and not gpu,
    cpu_count = cpu,
    gpu = gpu_name,
    gpu_count = gpu_count,
    workspace_id = workspace_id,
    project_name = inst.project_name,
    created_time = created_time,
  )
  py_f_data = {k:v for k,v in py_data.items() if v is not None}

  assets = U.join(U.folder(__file__), "assets")
  path = U.join(assets, "job_new.jinja")
  with open(path, "r") as f, open("exe.py", "w") as f2:
    f2.write(jinja2.Template(f.read()).render(**py_f_data))

  md_data = dict(
    project_name = project_name,
    created_time = created_time,
  )

  path = U.join(assets, "job_new_readme.jinja")
  with open(path, "r") as f, open("README.md", "w") as f2:
    f2.write(jinja2.Template(f.read()).render(**md_data))

def new(project_name, b: bool = False, workspace_id: str = None):
  """Create a new job, this can be run on NBX-Jobs or on an instance.

  Args:
    project_name (str): The name of the job folder
    b (bool, def. 'False'): If True, then job will run on an instance
    workspace_id (str, def. 'None'): If defined, that workspace will be used
      else personal workspace will be used
  """
  project_name = str(project_name)
  out = re.findall("^[a-zA-Z0-9_]+$", project_name)
  if not out:
    raise ValueError("Project name can only contain letters and underscore")

  if os.path.exists(project_name):
    raise ValueError(f"Project {project_name} already exists")

  fn = _build_job if b else _nbx_job
  fn(project_name, workspace_id)

  with open("requirements.txt", "w") as f:
    f.write(f"nbox=={__version__}")

  logger.debug("Completed")

def get_job_list(workspace_id: str = None):
  """Get list of jobs, optionally in a workspace"""
  auth_info = NBXAuthInfo(workspace_id = workspace_id)
  out: ListJobsResponse = rpc(
    nbox_grpc_stub.ListJobs,
    ListJobsRequest(auth_info = auth_info),
    "Could not get job list",
  )

  out = message_to_dict(out)
  if len(out["Jobs"]) == 0:
    logger.info("No jobs found")
    sys.exit(0)

  # filters = [f.upper() for f in filters]
  headers=list(out["Jobs"][0].keys())
  data = []
  for j in out["Jobs"]:
    _row = []
    for x in headers:
      if x == "status":
        _row.append(JobProto.Status.keys()[j[x]])
        continue
      _row.append(j[x])
      # if "*" in filters:
      #   _row.append(j[x])
      # if j["status"] in filters:
      #   _row.append(j[x])
    data.append(_row)
  for l in tabulate.tabulate(data, headers).splitlines():
    logger.info(l)


################################################################################
# NimbleBox.ai Jobs
# =================
# This is the actual job object that users can manipulate. It is a shallow class
# around the NBX-Jobs gRPC API.
################################################################################

class Job:
  new = staticmethod(new)
  status = staticmethod(get_job_list)

  def __init__(self, id, workspace_id = None):
    """Python wrapper for NBX-Jobs gRPC API

    Args:
        id (str): job ID
        workspace_id (str, optional): If None personal workspace is used. Defaults to None.
    """
    self.id = id
    self.workspace_id = workspace_id
    self.job_proto = JobProto(id = id, auth_info = NBXAuthInfo(workspace_id = workspace_id))
    self.job_info = JobInfo(job=self.job_proto)
    self.refresh()

  def change_schedule(self, new_schedule: 'Schedule'):
    """Change schedule this job"""
    logger.info(f"Updating job '{self.job_proto.id}'")
    self.job_proto.schedule.MergeFrom(new_schedule.get_message())
    rpc(
      nbox_grpc_stub.UpdateJob,
      UpdateJobRequest(job=self.job_proto, update_mask=FieldMask(paths=["schedule"])),
      "Could not update job schedule",
      raise_on_error = True
    )
    logger.info(f"Updated job '{self.job_proto.id}'")
    self.refresh()

  def __repr__(self) -> str:
    x = f"nbox.Job('{self.job_proto.id}', '{self.job_proto.auth_info.workspace_id}'): {self.status}"
    if self.job_proto.schedule.ByteSize != None:
      x += f" {self.job_proto.schedule}"
    else:
      x += " (no schedule)"
    return x

  def logs(self, f = sys.stdout):
    """Stream logs of the job, ``f`` can be anything has a ``.write/.flush`` methods"""
    logger.info(f"Streaming logs of job '{self.job_proto.id}'")
    for job_log in streaming_rpc(
      nbox_grpc_stub.GetJobLogs,
      JobLogsRequest(job = JobInfo(job = self.job_proto)),
      f"Could not get logs of job {self.job_proto.id}, is your job complete?",
      True
    ):
      for log in job_log.log:
        f.write(log + "\n")
        f.flush()

  def delete(self):
    """Delete this job"""
    logger.info(f"Deleting job '{self.job_proto.id}'")
    rpc(nbox_grpc_stub.DeleteJob, JobInfo(job = self.job_proto,), "Could not delete job")
    logger.info(f"Deleted job '{self.job_proto.id}'")
    self.refresh()

  def refresh(self):
    """Refresh Job statistics"""
    logger.info(f"Updating job '{self.job_proto.id}'")
    self.job_proto: JobProto = rpc(
      nbox_grpc_stub.GetJob, JobInfo(job = self.job_proto), f"Could not get job {self.job_proto.id}"
    )
    self.job_proto.auth_info.CopyFrom(NBXAuthInfo(workspace_id = self.workspace_id))
    self.job_info.CopyFrom(JobInfo(job = self.job_proto))
    logger.info(f"Updated job '{self.job_proto.id}'")

    self.status = self.job_proto.Status.keys()[self.job_proto.status]

  def trigger(self):
    """Manually triger this job"""
    logger.info(f"Triggering job '{self.job_proto.id}'")
    rpc(nbox_grpc_stub.TriggerJob, JobInfo(job=self.job_proto), f"Could not trigger job '{self.job_proto.id}'")
    logger.info(f"Triggered job '{self.job_proto.id}'")
    self.refresh()

  def __call__(self):
    pass

  def pause(self):
    """Pause the execution of this job.
    
    WARNING: This will remove all the scheduled runs, if present""" 
    logger.info(f"Pausing job '{self.job_proto.id}'")
    job: JobProto = self.job_proto
    job.status = JobProto.Status.PAUSED
    rpc(nbox_grpc_stub.UpdateJob, UpdateJobRequest(job=job, update_mask=FieldMask(paths=["status", "paused"])), f"Could not pause job {self.job_proto.id}", True)
    logger.info(f"Paused job '{self.job_proto.id}'")
    self.refresh()
  
  def resume(self):
    """Resume the Job with the current schedule, if provided else simlpy sets status as ACTIVE"""
    logger.info(f"Resuming job '{self.job_proto.id}'")
    job: JobProto = self.job_proto
    job.status = JobProto.Status.SCHEDULED
    rpc(nbox_grpc_stub.UpdateJob, UpdateJobRequest(job=job, update_mask=FieldMask(paths=["status", "paused"])), f"Could not resume job {self.job_proto.id}", True)
    logger.info(f"Resumed job '{self.job_proto.id}'")
    self.refresh()
