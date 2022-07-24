"""
``nbox.Job`` is a wrapper to the APIs that's it.

Notes
-----

* ``datetime.now(timezone.utc)`` is incorrect, use `this <https://blog.ganssle.io/articles/2019/11/utcnow.html>`_ method.
"""

import os
import sys
import jinja2
import tabulate
from functools import partial
from datetime import datetime, timedelta, timezone
from google.protobuf.timestamp_pb2 import Timestamp
from google.protobuf.field_mask_pb2 import FieldMask

import nbox.utils as U
from nbox.utils import logger
from nbox.auth import secret
from nbox.version import __version__
from nbox.messages import rpc, streaming_rpc
from nbox.hyperloop.nbox_ws_pb2 import JobInfo
from nbox.init import nbox_grpc_stub, nbox_ws_v1

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

  from nbox import Operator
  from nbox.network import deploy_job, deploy_serving
  operator: Operator = get_op(method == "serving")

  if method == "jobs":
    out: Job = deploy_job(
      init_folder = init_folder,
      job_id_or_name = id_or_name,
      dag = operator._get_dag(),
      workspace_id = workspace_id,
      schedule = get_schedule(),
      resource = get_resource(),
      _unittest = False
    )
    return out
  elif method == "serving":
    out: Serve = deploy_serving(
      init_folder = init_folder,
      deployment_id_or_name = id_or_name,
      workspace_id = workspace_id,
      resource = get_resource(),
      wait_for_deployment = False,
      _unittest = False
    )
  
  return out

################################################################################
"""
# NimbleBox.ai Serving

This is the proposed interface for the NimbleBox.ai Serving API. We want to keep
the highest levels of consistency with the NBX-Jobs API.
"""
################################################################################

def _get_deployment_data(id_or_name, workspace_id):
  # filter and get "id" and "name"
  stub_all_depl = nbox_ws_v1.workspace.u(workspace_id).deployments
  all_deployments = stub_all_depl()
  models = list(filter(lambda x: x["deployment_id"] == id_or_name or x["deployment_name"] == id_or_name, all_deployments))
  if len(models) == 0:
    logger.info(f"No Job found with ID or name: {id_or_name}, will create a new one")
    serving_name = id_or_name
    serving_id = None
  elif len(models) > 1:
    raise ValueError(f"Multiple models found for '{id_or_name}', try passing ID")
  else:
    logger.info(f"Found deployment with ID or name: {id_or_name}, will update it")
    data = models[0]
    serving_name = data["deployment_name"]
    serving_id = data["deployment_id"]
  return serving_id, serving_name


def print_serving_list(workspace_id: str = None, sort: str = "created_on"):
  def _get_time(t):
    return datetime.fromtimestamp(int(float(t))).strftime("%Y-%m-%d %H:%M:%S")

  stub_all_depl = nbox_ws_v1.workspace.u(workspace_id).deployments
  all_deployments = stub_all_depl()
  sorted_depls = sorted(all_deployments, key = lambda x: x[sort], reverse = sort == "created_on")
  headers = ["created_on", "id", "name", "pinned_id", "pinned_name", "pinned_last_updated"]
  all_depls = []
  for depl in sorted_depls:
    _depl = (_get_time(depl["created_on"]), depl["deployment_id"], depl["deployment_name"],)
    pinned = depl["pinned_model"]
    if not pinned:
      _depl += (None, None,)
    else:
      _depl += (pinned["id"], pinned["name"], _get_time(pinned["last_updated"]),)
    all_depls.append(_depl)

  for l in tabulate.tabulate(all_depls, headers).splitlines():
    logger.info(l)


def serving_forward(id_or_name: str, token: str, workspace_id: str = None, **kwargs):
  if workspace_id is None:
    raise DeprecationWarning("Personal workspace does not support serving")
  from nbox.operator import Operator
  op = Operator.from_serving(id_or_name, token, workspace_id)
  out = op(**kwargs)
  logger.info(out)


class Serve:
  new = staticmethod(new) # create a new folder, alias but `jobs new` should be used
  status = staticmethod(print_serving_list)
  upload = staticmethod(partial(upload_job_folder, "serving"))
  forward = staticmethod(serving_forward)

  def __init__(self, id, workspace_id = None) -> None:
    """Python wrapper for NBX-Serving gRPC API

    Args:
      id (str): Deployment ID
      workspace_id (str, optional): If None personal workspace is used. Defaults to None.
    """
    self.id = id
    self.workspace_id = workspace_id
    if workspace_id is None:
      raise DeprecationWarning("Personal workspace does not support serving")
    else:
      serving_id, serving_name = _get_deployment_data(self.id, self.workspace_id)
    self.serving_id = serving_id
    self.serving_name = serving_name
    self.ws_stub = nbox_ws_v1.workspace.u(workspace_id).deployments
    
  # def change_behaviour(self, new_behaviour: 'Behaviour' = None):
  #   raise NotImplementedError("Not implemented yet")

  def __repr__(self) -> str:
    x = f"nbox.Serve('{self.id}', '{self.workspace_id}')"
    return x

   

################################################################################
"""
# NimbleBox.ai Jobs

This is the actual job object that users can manipulate. It is a shallow class
around the NBX-Jobs gRPC API.
"""
################################################################################

def _get_job_data(id_or_name, workspace_id):
  # get stub
  if workspace_id == None:
    stub_all_jobs = nbox_ws_v1.user.jobs
  else:
    stub_all_jobs = nbox_ws_v1.workspace.u(workspace_id).jobs

  # filter and get "id" and "name"
  all_jobs = stub_all_jobs()
  jobs = list(filter(lambda x: x["job_id"] == id_or_name or x["name"] == id_or_name, all_jobs))
  if len(jobs) == 0:
    job_name =  id_or_name
    job_id = None
    logger.info(f"No Job found with ID or name: {job_name}")
  elif len(jobs) > 1:
    raise ValueError(f"Multiple jobs found for '{id_or_name}', try passing ID")
  else:
    data = jobs[0]
    job_name = data["name"]
    job_id = data["job_id"]
    logger.info(f"Found job with ID '{job_id}' and name '{job_name}'")
  
  return job_id, job_name


def get_job_list(workspace_id: str = None, sort: str = "name"):
  """Get list of jobs, optionally in a workspace"""
  def _get_time(t):
    return datetime.fromtimestamp(int(float(t))).strftime("%Y-%m-%d %H:%M:%S")

  auth_info = NBXAuthInfo(workspace_id = workspace_id)
  out: ListJobsResponse = rpc(
    nbox_grpc_stub.ListJobs,
    ListJobsRequest(auth_info = auth_info),
    "Could not get job list",
  )

  if len(out.Jobs) == 0:
    logger.info("No jobs found")
    sys.exit(0)

  headers = ['created_at', 'id', 'name', 'schedule', 'status']
  try:
    sorted_jobs = sorted(out.Jobs, key = lambda x: getattr(x, sort))
  except:
    logger.error(f"Cannot sort on key: {sort}")
  data = []
  for j in sorted_jobs:
    _row = []
    for x in headers:
      if x == "status":
        _row.append(JobProto.Status.keys()[getattr(j, x)])
        continue
      elif x == "created_at":
        _row.append(_get_time(j.created_at.seconds))
      elif x == "schedule":
        _row.append(j.schedule.cron)
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
    self.id, self.name = _get_job_data(id, workspace_id)
    self.workspace_id = workspace_id
    self.job_proto = JobProto(id = self.id, auth_info = NBXAuthInfo(workspace_id = workspace_id))
    self.job_info = JobInfo(job = self.job_proto)

    self.run_stub = None
    self.runs = []

    # sometimes a Job can be a placeholder and not have any data against it, thus it won't make
    # sense to try to get the data. This causes RPC error 
    if self.id is not None:
      self.refresh()
      self.get_runs() # load the runs as well

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
    if self.id == None:
      self.id, self.name = _get_job_data(self.id, self.workspace_id)
    if self.id == None:
      return
      
    self.job_proto: JobProto = rpc(
      nbox_grpc_stub.GetJob, JobInfo(job = self.job_proto), f"Could not get job {self.job_proto.id}"
    )
    self.job_proto.auth_info.CopyFrom(NBXAuthInfo(workspace_id = self.workspace_id))
    self.job_info.CopyFrom(JobInfo(job = self.job_proto))
    logger.debug(f"Updated job '{self.job_proto.id}'")

    self.status = self.job_proto.Status.keys()[self.job_proto.status]

  def trigger(self):
    """Manually triger this job"""
    logger.debug(f"Triggering job '{self.job_proto.id}'")
    rpc(nbox_grpc_stub.TriggerJob, JobInfo(job=self.job_proto), f"Could not trigger job '{self.job_proto.id}'")
    logger.info(f"Triggered job '{self.job_proto.id}'")
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

  def get_runs(self, sort = "s_no", reverse = False):
    self.run_stub = nbox_ws_v1.workspace.u(self.workspace_id).job.u(self.id).runs
    self.runs = self.run_stub()["runs_list"]
    sorted_runs = sorted(self.runs, key = lambda x: x[sort], reverse = reverse)
    return sorted_runs

  def display_runs(self, sort: str = "created_at", n: int = -1):
    runs = self.get_runs(sort, sort == "created_at") # time should be reverse
    if n == -1:
      n = len(runs)
    headers = ["s_no", "created_at", "run_id", "status"]
    data = []
    for i, run in enumerate(runs):
      if i >= n:
        break
      data.append((run["s_no"], run["created_at"], run["run_id"], run["status"]))
    for l in tabulate.tabulate(data, headers).splitlines():
      logger.info(l)

class Schedule:
  def __init__(
    self,
    hour: int  = None,
    minute: int = None,
    days: list = [],
    months: list = [],
    starts: datetime = None,
    ends: datetime = None,
  ):
    """Make scheduling natural.

    Usage:

    .. code-block:: python

      # 4:20 everyday
      Schedule(4, 0)

      # 4:20 every friday
      Schedule(4, 20, ["fri"])

      # 4:20 every friday from jan to feb
      Schedule(4, 20, ["fri"], ["jan", "feb"])

      # 4:20 everyday starting in 2 days and runs for 3 days
      starts = datetime.now(timezone.utc) + timedelta(days = 2) # NOTE: that time is in UTC
      Schedule(4, 20, starts = starts, ends = starts + timedelta(days = 3))

      # Every 1 hour
      Schedule(1)

      # Every 69 minutes
      Schedule(minute = 69)

    Args:
        hour (int): Hour of the day, if only this value is passed it will run every ``hour``
        minute (int): Minute of the hour, if only this value is passed it will run every ``minute``
        days (list, optional): List of days (first three chars) of the week, if not passed it will run every day.
        months (list, optional): List of months (first three chars) of the year, if not passed it will run every month.
        starts (datetime, optional): UTC Start time of the schedule, if not passed it will start now.
        ends (datetime, optional): UTC End time of the schedule, if not passed it will end in 7 days.
    """
    self.hour = hour
    self.minute = minute

    self._is_repeating = self.hour or self.minute
    self.mode = None
    if self.hour == None and self.minute == None:
      raise ValueError("Atleast one of hour or minute should be passed")
    elif self.hour != None and self.minute != None:
      assert self.hour in list(range(0, 24)), f"Hour must be in range 0-23, got {self.hour}"
      assert self.minute in list(range(0, 60)), f"Minute must be in range 0-59, got {self.minute}"
    elif self.hour != None:
      assert self.hour in list(range(0, 24)), f"Hour must be in range 0-23, got {self.hour}"
      self.mode = "every"
      self.minute = datetime.now(timezone.utc).strftime("%m") # run every this minute past this hour
      self.hour = f"*/{self.hour}"
    elif self.minute != None:
      self.hour = self.minute // 60
      assert self.hour in list(range(0, 24)), f"Hour must be in range 0-23, got {self.hour}"
      self.minute = f"*/{self.minute % 60}"
      self.hour = f"*/{self.hour}" if self.hour > 0 else "*"
      self.mode = "every"

    _days = {k:str(i) for i,k in enumerate(["sun","mon","tue","wed","thu","fri","sat"])}
    _months = {k:str(i+1) for i,k in enumerate(["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"])}

    diff = set(days) - set(_days.keys())
    if len(diff):
      raise ValueError(f"Invalid days: {diff}")
    self.days = ",".join([_days[d] for d in days]) if days else "*"

    diff = set(months) - set(_months.keys())
    if len(diff):
      raise ValueError(f"Invalid months: {diff}")
    self.months = ",".join([_months[m] for m in months]) if months else "*"

    self.starts = starts or datetime.now(timezone.utc)
    self.ends = ends or datetime.now(timezone.utc) + timedelta(days = 7)

  @property
  def cron(self):
    """Cron string"""
    if self.mode == "every":
      return f"{self.minute} {self.hour} * * *"
    return f"{self.minute} {self.hour} * {self.months} {self.days}"

  def get_dict(self):
    return {"cron": self.cron, "mode": self.mode, "starts": self.starts, "ends": self.ends}

  def get_message(self) -> JobProto.Schedule:
    """Get the JobProto.Schedule object for this Schedule"""
    _starts = Timestamp(); _starts.GetCurrentTime()
    _ends = Timestamp(); _ends.FromDatetime(self.ends)
    # return JobProto.Schedule(start = _starts, end = _ends, cron = self.cron)
    return JobProto.Schedule(cron = self.cron)

  def __repr__(self):
    return str(self.get_dict())
