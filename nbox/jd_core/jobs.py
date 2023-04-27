import sys
from datetime import datetime
import tabulate
from functools import lru_cache
from typing import Tuple, List, Dict, Any

from google.protobuf.field_mask_pb2 import FieldMask

from nbox import messages as mpb
from nbox.utils import logger, lo
from nbox.auth import secret, auth_info_pb
from nbox.init import nbox_ws_v1, nbox_grpc_stub

from nbox.hyperloop.jobs.nbox_ws_pb2 import (
  JobRequest,
  ListJobsRequest,
  ListJobsResponse,
  UpdateJobRequest
)
from nbox.hyperloop.jobs.job_pb2 import Job as JobProto

from nbox.jd_core.schedule import Schedule

################################################################################
"""
# NimbleBox.ai Jobs

This is the actual job object that users can manipulate. It is a shallow class
around the NBX-Jobs gRPC API.
"""
################################################################################

@lru_cache()
def _get_job_data(name: str = "", id: str = "", remove_archived: bool = True, *, workspace_id: str = "") -> Tuple[str, str]:
  """
  Returns the job id and name
  
  Args:
    name (str, optional): Job name. Defaults to "".
    id (str, optional): Job ID. Defaults to "".
    remove_archived (bool, optional): If True, will remove archived jobs. Defaults to True.

  Returns:
    Tuple[str, str]: Job ID, Job Name
  """
  if (not name and not id) or (name and id):
    logger.info(f"Please either pass job_id '{id}' or name '{name}'")
    return None, None
  # get stub
  workspace_id = workspace_id or secret.workspace_id
  if workspace_id == None:
    workspace_id = "personal"

  job: JobProto = mpb.rpc(
    nbox_grpc_stub.GetJob,
    JobRequest(
      auth_info = auth_info_pb(),
      job = JobProto(id=id, name=name)
    ),
    "Could not find job with ID: {}".format(id),
    raise_on_error = True
  )
  if job.status == JobProto.Status.ARCHIVED and remove_archived:
    logger.info(f"Job {job.id} is archived. Will try to create a new Job.")
    return None, name
  job_name = job.name
  job_id = job.id
  logger.info(f"Found job with ID '{job_id}' and name '{job_name}'")
  return job_id, job_name

def get_job_list(sort: str = "created_at", *, workspace_id: str = ""):
  """Get list of jobs
  
  Args:
    sort (str, optional): Sort key. Defaults to "name".
  """
  workspace_id = workspace_id or secret.workspace_id

  def _get_time(t):
    return datetime.fromtimestamp(int(float(t))).strftime("%Y-%m-%d %H:%M:%S")

  out: ListJobsResponse = mpb.rpc(
    nbox_grpc_stub.ListJobs,
    ListJobsRequest(auth_info = auth_info_pb()),
    "Could not get job list",
  )

  if len(out.jobs) == 0:
    logger.info("No jobs found")
    sys.exit(0)

  headers = ['created_at', 'id', 'name', 'schedule', 'status']
  try:
    sorted_jobs = sorted(out.jobs, key = lambda x: getattr(x, sort))
  except Exception as e:
    logger.error(f"Cannot sort on key: {sort}")
    sorted_jobs = out.jobs
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

  def __init__(self, job_name: str = "", job_id: str = ""):
    """Python wrapper for NBX-Jobs gRPC API, when both arguments are not passed,
    an unintialiased object is created.

    Args:
      job_name (str, optional): Job name. Defaults to "".
      job_id (str, optional): Job ID. Defaults to "".
    """
    if job_name == "" and job_id == "":
      return

    self.id, self.name = _get_job_data(job_name, job_id)
    self.workspace_id = secret.workspace_id
    self.auth_info = auth_info_pb()
    self.job_proto = JobProto(id = self.id)

    self.run_stub = None
    self.runs = []

    # sometimes a Job can be a placeholder and not have any data against it, thus it won't make
    # sense to try to get the data. This causes RPC error
    if self.id is not None:
      self.refresh()
      self.get_runs() # load the runs as well

  @property
  def exists(self):
    """Check if this job exists in the workspace"""
    return self.id is not None

  @classmethod
  def new(cls, job_name: str, description: str = ""):
    """Create a new job

    Args:
      job_name (str): Job name
      description (str, optional): Job description. Defaults to "".

    Returns:
      Job: Job object
    """
    job = nbox_ws_v1.job("post", job_name = job_name, job_description = description)
    job_id = job["job_id"]
    # job_proto: JobProto = nbox_grpc_stub.CreateJob(
    #   JobRequest(
    #     auth_info = auth_info_pb(),
    #     job = JobProto(
    #       name = job_name,
    #       code = Code(
    #         size = 1,
    #         type = Code.Type.NOT_SET,
    #       ),
    #       resource = DEFAULT_RESOURCE,
    #       schedule = JobProto.Schedule(
    #         cron = "0 0 * * *",
    #         end = Timestamp(seconds = 0),
    #       ),
    #       status = JobProto.Status.NOT_SET,
    #       paused = False,
    #     ),
    #   )
    # )
    logger.info(f"Created a new job with name (ID): {job_name} ({job_id})")
    return cls(job_id = job_id)

  def change_schedule(self, new_schedule: Schedule):
    """Change schedule this job
    
    Args:
      new_schedule (Schedule): New schedule
    """
    logger.debug(f"Updating job '{self.job_proto.id}'")
    self.job_proto.schedule.MergeFrom(new_schedule.get_message())
    mpb.rpc(
      nbox_grpc_stub.UpdateJob,
      UpdateJobRequest(auth_info=self.auth_info, job=self.job_proto, update_mask=FieldMask(paths=["schedule"])),
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
    """Stream logs of the job, `f` can be anything has a `.write/.flush` methods"""
    logger.debug(f"Streaming logs of job '{self.job_proto.id}'")
    for job_log in mpb.streaming_rpc(
      nbox_grpc_stub.GetJobLogs,
      JobRequest(auth_info=self.auth_info ,job = self.job_proto),
      f"Could not get logs of job {self.job_proto.id}, is your job complete?",
      True
    ):
      for log in job_log.log:
        f.write(log)
        f.flush()

  def delete(self):
    """Delete this job"""
    logger.info(f"Deleting job '{self.job_proto.id}'")
    mpb.rpc(nbox_grpc_stub.DeleteJob, JobRequest(auth_info=self.auth_info, job = self.job_proto,), "Could not delete job")
    logger.info(f"Deleted job '{self.job_proto.id}'")
    self.refresh()

  def refresh(self):
    """Refresh Job data"""
    logger.info(f"Updating job '{self.job_proto.id}'")
    if self.id == None:
      self.id, self.name = _get_job_data(id = self.id, workspace_id = self.workspace_id)
    if self.id == None:
      return

    self.job_proto: JobProto = mpb.rpc(
      nbox_grpc_stub.GetJob,
      JobRequest(auth_info=self.auth_info, job = self.job_proto),
      f"Could not get job {self.job_proto.id}"
    )
    self.auth_info.CopyFrom(auth_info_pb())
    logger.debug(f"Updated job '{self.job_proto.id}'")

    self.status = self.job_proto.Status.keys()[self.job_proto.status]

  def trigger(self, tag: str = "", feature_gates: dict = {}):
    """Manually triger this job.
    
    Args:
      tag (str, optional): Tag to be set in the run metadata, read in more detail before trying to use this. Defaults to "".
    """
    logger.debug(f"Triggering job '{self.job_proto.id}'")
    if tag:
      self.job_proto.feature_gates.update(feature_gates)
      self.job_proto.feature_gates.update({"SetRunMetadata": tag})
    mpb.rpc(nbox_grpc_stub.TriggerJob, JobRequest(auth_info=self.auth_info, job = self.job_proto), f"Could not trigger job '{self.job_proto.id}'")
    logger.info(f"Triggered job '{self.job_proto.id}'")
    self.refresh()

  def pause(self):
    """Pause the execution of this job.

    **WARNING: This will "cancel" all the scheduled runs, if present**
    """
    logger.info(f"Pausing job '{self.job_proto.id}'")
    job: JobProto = self.job_proto
    job.status = JobProto.Status.PAUSED
    mpb.rpc(nbox_grpc_stub.UpdateJob, UpdateJobRequest(auth_info=self.auth_info, job=job, update_mask=FieldMask(paths=["status", "paused"])), f"Could not pause job {self.job_proto.id}", True)
    logger.debug(f"Paused job '{self.job_proto.id}'")
    self.refresh()

  def resume(self):
    """Resume the Job with the current schedule, if provided else simlpy sets status as ACTIVE"""
    logger.info(f"Resuming job '{self.job_proto.id}'")
    job: JobProto = self.job_proto
    job.status = JobProto.Status.SCHEDULED
    mpb.rpc(nbox_grpc_stub.UpdateJob, UpdateJobRequest(auth_info=self.auth_info, job=job, update_mask=FieldMask(paths=["status", "paused"])), f"Could not resume job {self.job_proto.id}", True)
    logger.debug(f"Resumed job '{self.job_proto.id}'")
    self.refresh()

  def _get_runs(self, page = -1, limit = 10) -> List[Dict]:
    self.run_stub = nbox_ws_v1.job.u(self.id).runs
    runs = self.run_stub(limit = limit, page = page)["runs_list"]
    return runs

  def get_runs(self, page = -1, sort = "s_no", limit = 10) -> List[Dict]:
    """
    Get runs for this job

    Args:
      page (int, optional): Page number. Defaults to -1.
      sort (str, optional): Sort by. Defaults to "s_no".
      limit (int, optional): Number of runs to return. Defaults to 10.

    Returns:
      List[Dict]: List of runs
    """
    runs = self._get_runs(page, limit)
    sorted_runs = sorted(runs, key = lambda x: x[sort])
    return sorted_runs

  def display_runs(self, sort: str = "created_at", page: int = -1, limit = 10):
    """Display runs for this job

    Args:
      sort (str, optional): Sort by. Defaults to "created_at".
      page (int, optional): Page number. Defaults to -1.
      limit (int, optional): Number of runs to return. Defaults to 10.
    """
    headers = ["s_no", "created_at", "run_id", "status"]

    def _display_runs(runs):
      data = []
      for run in runs:
        data.append((run["s_no"], run["created_at"], run["run_id"], run["status"]))
      for l in tabulate.tabulate(data, headers).splitlines():
        logger.info(l)

    runs = self.get_runs(sort = sort, page = page, limit = limit) # time should be reverse
    _display_runs(runs)
    if page == -1:
      page = 1
    y = input(f">> Print {limit} more runs? (y/n): ")
    done = y != "y"
    while not done:
      _display_runs(self.get_runs(page = page + 1, sort = sort, limit = limit))
      page += 1
      y = input(f">> Print {limit} more runs? (y/n): ")
      done = y != "y"

  def last_n_runs(self, n: int = 10) -> List[Dict]:
    """Get last n runs for this job

    Args:
      n (int, optional): Number of runs to return. Defaults to 10.

    Returns:
      List[Dict]: List of runs
    """
    all_items = []
    _page = 1
    out = self._get_runs(_page, n)
    all_items.extend(out)

    while len(all_items) < n:
      _page += 1
      out = self._get_runs(_page, n)
      if not len(out):
        break
      all_items.extend(out)

    if n == 1:
      return all_items[0]
    return all_items
