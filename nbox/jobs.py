"""
``nbox.Job`` is a wrapper to the APIs that's it.

Notes
-----

* ``datetime.now(timezone.utc)`` is incorrect, use `this <https://blog.ganssle.io/articles/2019/11/utcnow.html>`_ method.
"""

import os
import sys
import json
import jinja2
import tabulate
import requests
from functools import partial
from datetime import datetime, timezone
from google.protobuf.field_mask_pb2 import FieldMask

import nbox.utils as U
from nbox.utils import logger
from nbox.auth import secret
from nbox.version import __version__
from nbox.hyperloop.nbox_ws_pb2 import JobInfo
from nbox.init import nbox_grpc_stub, nbox_ws_v1
from nbox.network import _get_deployment_data
from nbox.messages import message_to_dict, rpc, streaming_rpc
from nbox.hyperloop.job_pb2 import NBXAuthInfo, Job as JobProto
from nbox.hyperloop.nbox_ws_pb2 import ListJobsRequest, JobLogsRequest, ListJobsResponse, UpdateJobRequest


################################################################################
"""
# Common Functions

These functions are common to both NBX-Jobs and NBX-Deploy.
"""
################################################################################

def _repl_schedule(return_proto: bool = False):
  """Create a schedule from the user input.

  Args:
    return_proto (bool, optional): If `True` returns proto otherwise returns cron string.
  """
  from .network import Schedule
  logger.info("Calendar Instructions (all time is in UTC Timezone):")
  logger.info("            What you want: Code")
  logger.info("\"   every 4:30 at friday\": Schedule(4, 30, ['fri'])")
  logger.info("\"every 12:00 on weekends\": Schedule(12, 0, ['sat', 'sun'])")
  logger.info("\"         every 10 hours\": Schedule(10)")
  logger.info("\"      every 420 minutes\": Schedule(minute = 420)")
  logger.info("> Enter the calender instruction: ")
  cron_instr = input("> ").strip()
  schedule_proto = None
  try:
    schedule_proto: Schedule = eval(cron_instr, {'Schedule': Schedule})
  except Exception as e:
    logger.error(f"Invalid calender instruction: {e}")
    sys.exit(1)
  if not return_proto:
    return cron_instr
  return schedule_proto

def new(folder_name):
  """This creates a single folder that can be used with both NBX-Jobs and NBX-Deploy.

  Args:
    folder_name (str): The name of the job folder
  """
  folder_name = str(folder_name)
  if folder_name == "":
    raise ValueError("Folder name can not be empty")
  if os.path.exists(folder_name):
    raise ValueError(f"Project {folder_name} already exists")
  os.makedirs(folder_name)

  # get a timestamp like this: Monday W34 [UTC 12 April, 2022 - 12:00:00]
  _ct = datetime.now(timezone.utc)
  _day = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][_ct.weekday()]
  created_time = f"{_day} W{_ct.isocalendar()[1]} [ UTC {_ct.strftime('%d %b, %Y - %H:%M:%S')} ]"

  # we no longer need to ask for the deployment ID / Job ID since deployment is now concern of
  # nbx [jobs/serve] upload ...

  # create the necessary files that can be used
  os.chdir(folder_name)
  with open(".nboxignore", "w") as f:
    f.write("__pycache__/")
  
  with open("requirements.txt", "w") as f:
    f.write(f"nbox[serving]=={__version__} # do not change this")

  assets = U.join(U.folder(__file__), "assets")
  path = U.join(assets, "user.jinja")
  with open(path, "r") as f, open("nbx_user.py", "w") as f2:
    f2.write(jinja2.Template(f.read()).render({
      "created_time": created_time,
      "nbx_username": secret.get("username"),
    }))

  logger.info(f"Created folder: {folder_name}")

def upload_job_folder(method: str, init_folder: str, id_or_name: str, workspace_id: str = None):
  """Upload the code for a job to the NBX-Jobs if not present, it will create a new Job.

  Args:
    init_folder (str): folder with all the relevant files
    id_or_name (str): Job ID or name
    workspace_id (str, optional): Workspace ID, `None` for personal workspace. Defaults to None.
  """
  sys.path.append(init_folder)
  from nbx_user import get_op, get_resource, get_schedule

  operator = get_op()

  if method == "jobs":
    operator.deploy(
      init_folder = init_folder,
      job_id_or_name = id_or_name,
      workspace_id = workspace_id,
      schedule = get_schedule(),
      resource = get_resource(),
      _unittest = False
    )
  elif method == "serving":
    operator.serve(
      init_folder = init_folder,
      id_or_name = id_or_name,
      workspace_id = workspace_id,
      resource = get_resource(),
      wait_for_deployment = False,
    )
   

################################################################################
"""
# NimbleBox.ai Jobs

This is the actual job object that users can manipulate. It is a shallow class
around the NBX-Jobs gRPC API.
"""
################################################################################

def get_job_list(workspace_id: str = None, sort: str = "name"):
  """Get list of jobs, optionally in a workspace"""
  auth_info = NBXAuthInfo(workspace_id = workspace_id)
  out: ListJobsResponse = rpc(
    nbox_grpc_stub.ListJobs,
    ListJobsRequest(auth_info = auth_info),
    "Could not get job list",
  )

  if len(out.Jobs) == 0:
    logger.info("No jobs found")
    sys.exit(0)

  headers = [x[0].name for x in out.Jobs[0].ListFields()]
  sorted_jobs = sorted(out.Jobs, key = lambda x: getattr(x, sort))
  data = []
  for j in sorted_jobs:
    _row = []
    for x in headers:
      if x == "status":
        _row.append(JobProto.Status.keys()[getattr(j, x)])
        continue
      elif x == "created_at":
        _row.append(datetime.fromtimestamp(getattr(j, x).seconds).strftime("%d %b, %Y - %H:%M"))
      else:
        _row.append(getattr(j, x))
    data.append(_row)
  for l in tabulate.tabulate(data, headers).splitlines():
    logger.info(l)

class Job:
  new = staticmethod(new)
  status = staticmethod(get_job_list)
  upload = staticmethod(partial(upload_job_folder, "jobs"))

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

  def change_schedule(self, new_schedule: 'Schedule' = None):
    """Change schedule this job"""
    logger.debug(f"Updating job '{self.job_proto.id}'")
    if new_schedule == None:
      new_schedule = _repl_schedule(True) # get the information from REPL
    self.job_proto.schedule.MergeFrom(new_schedule.get_message())
    rpc(
      nbox_grpc_stub.UpdateJob,
      UpdateJobRequest(job=self.job_proto, update_mask=FieldMask(paths=["schedule"])),
      "Could not update job schedule",
      raise_on_error = True
    )
    logger.debug(f"Updated job '{self.job_proto.id}'")
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
    logger.debug(f"Streaming logs of job '{self.job_proto.id}'")
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
    logger.debug(f"Updating job '{self.job_proto.id}'")
    self.job_proto: JobProto = rpc(
      nbox_grpc_stub.GetJob, JobInfo(job = self.job_proto), f"Could not get job {self.job_proto.id}"
    )
    self.job_proto.auth_info.CopyFrom(NBXAuthInfo(workspace_id = self.workspace_id))
    self.job_info.CopyFrom(JobInfo(job = self.job_proto))
    logger.debug(f"Updated job '{self.job_proto.id}'")

    self.status = self.job_proto.Status.keys()[self.job_proto.status]

  def trigger(self):
    """Manually triger this job"""
    logger.info(f"Triggering job '{self.job_proto.id}'")
    rpc(nbox_grpc_stub.TriggerJob, JobInfo(job=self.job_proto), f"Could not trigger job '{self.job_proto.id}'")
    logger.debug(f"Triggered job '{self.job_proto.id}'")
    self.refresh()

  def pause(self):
    """Pause the execution of this job.
    
    WARNING: This will "cancel" all the scheduled runs, if present""" 
    logger.info(f"Pausing job '{self.job_proto.id}'")
    job: JobProto = self.job_proto
    job.status = JobProto.Status.PAUSED
    rpc(nbox_grpc_stub.UpdateJob, UpdateJobRequest(job=job, update_mask=FieldMask(paths=["status", "paused"])), f"Could not pause job {self.job_proto.id}", True)
    logger.debug(f"Paused job '{self.job_proto.id}'")
    self.refresh()
  
  def resume(self):
    """Resume the Job with the current schedule, if provided else simlpy sets status as ACTIVE"""
    logger.info(f"Resuming job '{self.job_proto.id}'")
    job: JobProto = self.job_proto
    job.status = JobProto.Status.SCHEDULED
    rpc(nbox_grpc_stub.UpdateJob, UpdateJobRequest(job=job, update_mask=FieldMask(paths=["status", "paused"])), f"Could not resume job {self.job_proto.id}", True)
    logger.debug(f"Resumed job '{self.job_proto.id}'")
    self.refresh()

################################################################################
"""
# NimbleBox.ai Serving

This is the proposed interface for the NimbleBox.ai Serving API. We want to keep
the highest levels of consistency with the NBX-Jobs API.
"""
################################################################################

def get_serving_list(workspace_id: str = None, sort: str = "name"):
  raise NotImplementedError("Not implemented yet")


def serving_forward(id_or_name: str, token: str, workspace_id: str = None, **kwargs):
  if workspace_id is None:
    raise DeprecationWarning("Personal workspace does not support serving")
  from nbox.operator import Operator
  op = Operator.from_serving(id_or_name, token, workspace_id)
  out = op(**kwargs)
  logger.info(out)

class Serve:
  new = staticmethod(new) # create a new folder, alias but `jobs new` should be used
  status = staticmethod(get_serving_list)
  upload = staticmethod(partial(upload_job_folder, "serving"))
  forward = staticmethod(serving_forward)

  def __init__(self, id, workspace_id = None) -> None:
    """Python wrapper for NBX-Serving gRPC API

    Args:
      id (str): job ID
      workspace_id (str, optional): If None personal workspace is used. Defaults to None.
    """
    self.id = id
    self.workspace_id = workspace_id
    # self.serving_proto = ServingProto(id = id, auth_info = NBXAuthInfo(workspace_id = workspace_id))
    # self.serving_info = ServingInfo(job=self.job_proto)
    # self.refresh()

    if workspace_id is None:
      raise DeprecationWarning("Personal workspace does not support serving")
    else:
      serving_id, serving_name = _get_deployment_data(self.id, self.workspace_id)
    self.serving_id = serving_id
    self.serving_name = serving_name
    self.ws_stub = nbox_ws_v1.workspace.u(workspace_id).deployments
    
  def change_behaviour(self, new_behaviour: 'Behaviour' = None):
    raise NotImplementedError("Not implemented yet")

  def __repr__(self) -> str:
    x = f"nbox.Job('{self.job_proto.id}', '{self.job_proto.auth_info.workspace_id}'): {self.status}"
    if self.job_proto.schedule.ByteSize != None:
      x += f" {self.job_proto.schedule}"
    else:
      x += " (no schedule)"
    return x

  def logs(self, f = sys.stdout):
    """Stream logs of the job, ``f`` can be anything has a ``.write/.flush`` methods"""
    logger.debug(f"Streaming logs of job '{self.job_proto.id}'")
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
    logger.debug(f"Updating job '{self.job_proto.id}'")
    self.job_proto: JobProto = rpc(
      nbox_grpc_stub.GetJob, JobInfo(job = self.job_proto), f"Could not get job {self.job_proto.id}"
    )
    self.job_proto.auth_info.CopyFrom(NBXAuthInfo(workspace_id = self.workspace_id))
    self.job_info.CopyFrom(JobInfo(job = self.job_proto))
    logger.debug(f"Updated job '{self.job_proto.id}'")

    self.status = self.job_proto.Status.keys()[self.job_proto.status]

  def trigger(self):
    """Manually triger this job"""
    logger.info(f"Triggering job '{self.job_proto.id}'")
    rpc(nbox_grpc_stub.TriggerJob, JobInfo(job=self.job_proto), f"Could not trigger job '{self.job_proto.id}'")
    logger.debug(f"Triggered job '{self.job_proto.id}'")
    self.refresh()

  def pause(self):
    """Pause the execution of this job.
    
    WARNING: This will "cancel" all the scheduled runs, if present""" 
    logger.info(f"Pausing job '{self.job_proto.id}'")
    job: JobProto = self.job_proto
    job.status = JobProto.Status.PAUSED
    rpc(nbox_grpc_stub.UpdateJob, UpdateJobRequest(job=job, update_mask=FieldMask(paths=["status", "paused"])), f"Could not pause job {self.job_proto.id}", True)
    logger.debug(f"Paused job '{self.job_proto.id}'")
    self.refresh()
  
  def resume(self):
    """Resume the Job with the current schedule, if provided else simlpy sets status as ACTIVE"""
    logger.info(f"Resuming job '{self.job_proto.id}'")
    job: JobProto = self.job_proto
    job.status = JobProto.Status.SCHEDULED
    rpc(nbox_grpc_stub.UpdateJob, UpdateJobRequest(job=job, update_mask=FieldMask(paths=["status", "paused"])), f"Could not resume job {self.job_proto.id}", True)
    logger.debug(f"Resumed job '{self.job_proto.id}'")
    self.refresh()
