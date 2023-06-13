import sys
from datetime import datetime
import tabulate
from functools import lru_cache
from typing import Tuple, List, Dict, Any

from google.protobuf.field_mask_pb2 import FieldMask

from nbox import messages as mpb
from nbox.utils import logger, lo, DeprecationError
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
# functions
#
# helper functions that sit outside of a job
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

def get_job_list(limit: int = 20, offset: int = 0) -> ListJobsResponse:
  """Get list of jobs

  Args:
    limit (int, optional): Number of jobs to return. Defaults to 20.
    offset (int, optional): Offset. Defaults to 0.

  Returns:
    ListJobsResponse: List of jobs
  """
  out: ListJobsResponse = nbox_grpc_stub.ListJobs(ListJobsRequest(
    auth_info = auth_info_pb(),
    limit = limit,
    offset = offset,
  ))
  return out

def print_job_list(sort: str = "", limit = 20, offset = 0) -> None:
  """Pretty print list of jobs
  
  Args:
    limit (int, optional): Number of jobs to print. Defaults to 20.
    offset (int, optional): Offset. Defaults to 0.
  """

  #TODO: @yashbonde remove sorting
  if sort:
    logger.warning(f"Sorting will soon be deprecated")

  out = get_job_list(limit, offset)
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
        _row.append(datetime.fromtimestamp(int(float(j.created_at.seconds))).strftime("%Y-%m-%d %H:%M:%S"))
      elif x == "schedule":
        _row.append(j.schedule.cron)
      else:
        _row.append(getattr(j, x))
    data.append(_row)
  for l in tabulate.tabulate(data, headers).splitlines():
    logger.info(l)

def new_job(name: str, description: str = "") -> JobProto:
  """Create a new job

  Args:
    job_name (str): Job name
    description (str, optional): Job description. Defaults to "".

  Returns:
    Job: Job object
  """
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
  job = nbox_ws_v1.job("post", job_name = name, job_description = description, job_type = "", job_type_data = {})
  job_id = job["job_id"]
  logger.info(f"Created job with ID '{job_id}'")
  return Job(job_id = job_id)


################################################################################
# Main job class
#
# this contains all the things for a specific job
################################################################################

class Job:
  def __init__(self, job_id: str, job_name: str = ""):
    """Python wrapper for NBX-Jobs gRPC API, when both arguments are not passed,
    an unintialiased object is created.

    Args:
      job_name (str, optional): Job name. Defaults to "".
      job_id (str, optional): Job ID. Defaults to "".
    """
    if job_name:
      raise DeprecationError("pass job_id only")
    self.workspace_id = secret.workspace_id
    self.auth_info = auth_info_pb()
    self.job_pb: JobProto = nbox_grpc_stub.GetJob(JobRequest(auth_info=self.auth_info, job=JobProto(id = job_id)))
    self.run_stub = nbox_ws_v1.job.u(self.job_pb.id).runs
    self.runs = []

  def __repr__(self) -> str:
    x = f"nbox.Job('{self.job_pb.id}', {self.status})"
    if self.job_pb.schedule.ByteSize() != None:
      x += f" {self.job_pb.schedule}"
    return x

  #
  # getters
  #

  # things we want to keep

  @property
  def id(self):
    return self.job_pb.id
  
  @property
  def status(self):
    return self.job_pb.Status.keys()[self.job_pb.status]
  
  # things we want to get rid of

  @property
  def job_proto(self):
    logger.warning("job.job_proto is deprecated. Use job.job_pb instead")
    return self.job_pb

  @property
  def exists(self):
    """Check if this job exists in the workspace"""
    # raise DeprecationError("Deprecated")
    logger.warning("job.exists is deprecated, it should have never existed")

  #
  # APIs for talking to the RPC
  # Job.refresh          GetJob
  # Job.update_schedule  UpdateJob
  # Job.pause            UpdateJob
  # Job.resume           UpdateJob
  # Job.delete           DeleteJob
  # Job.trigger          TriggerJob
  # Job.logs             GetJobLogs
  #
  # get_job_list            ListJobs
  # network._upload_job_zip UploadJobCode
  #                         CreateJob
  #

  def refresh(self):
    """Refresh Job data"""
    logger.info(f"Updating job '{self.job_pb.id}'")
    if self.id == None:
      self.id, self.name = _get_job_data(id = self.id, workspace_id = self.workspace_id)
    if self.id == None:
      return

    self.job_pb = nbox_grpc_stub.GetJob(JobRequest(auth_info=self.auth_info, job = self.job_pb))
    self.auth_info.CopyFrom(auth_info_pb())
    logger.debug(f"Updated job '{self.job_pb.id}'")

  def update_schedule(self, new_schedule: Schedule):
    """Change schedule this job
    
    Args:
      new_schedule (Schedule): New schedule
    """
    self.job_pb.schedule.MergeFrom(new_schedule.get_message())
    self.job_pb: JobProto = nbox_grpc_stub.UpdateJob(UpdateJobRequest(
      auth_info=self.auth_info,
      job=self.job_pb,
      update_mask=FieldMask(paths=["schedule"])
    ))
    logger.info(f"Changed schedule of job '{self.job_pb.id}' to '{self.job_pb.schedule}'")

  def pause(self):
    """Pause the execution of this job.

    **WARNING: This will "cancel" all the scheduled runs, if present**
    """
    logger.info(f"Pausing job '{self.job_pb.id}'")
    job: JobProto = self.job_pb
    job.status = JobProto.Status.PAUSED
    self.job_pb = nbox_grpc_stub.UpdateJob(UpdateJobRequest(auth_info=self.auth_info, job=job, update_mask=FieldMask(paths=["status", "paused"])))
    logger.debug(f"Paused job '{self.job_pb.id}'")

  def resume(self):
    """Resume the Job with the current schedule, if provided else simlpy sets status as ACTIVE"""
    job: JobProto = self.job_pb
    job.status = JobProto.Status.SCHEDULED
    self.job_pb = nbox_grpc_stub.UpdateJob(UpdateJobRequest(auth_info=self.auth_info, job=job, update_mask=FieldMask(paths=["status", "paused"])))
    logger.info(f"Resumed job '{self.job_pb.id}'")

  def delete(self):
    """Delete this job"""
    logger.warning(f"Deleting job '{self.job_pb.id}'")
    nbox_grpc_stub.DeleteJob(JobRequest(auth_info=self.auth_info, job = self.job_pb))
    logger.warning(f"Deleted job '{self.job_pb.id}'")
    self.refresh()

  def trigger(self, tag: str = "", feature_gates: dict = {}):
    """Manually triger this job.

    Args:
      tag (str, optional): Tag to be set in the run metadata, read in more detail before trying to use this. Defaults to "".
      feature_gates (dict, optional): Feature gates to be set for this run. Defaults to {}.
    """
    if feature_gates:
      self.job_pb.feature_gates.update(feature_gates)
    if tag:
      self.job_pb.feature_gates.update({"SetRunMetadata": tag})
    nbox_grpc_stub.TriggerJob(JobRequest(auth_info=self.auth_info, job = self.job_pb))
    logger.info(f"Triggered job '{self.job_pb.id}'")

  def logs(self, f = sys.stdout):
    """Stream logs of the job, `f` can be anything has a `.write/.flush` methods"""
    logger.debug(f"Streaming logs of job '{self.job_pb.id}'")
    data_iter = nbox_grpc_stub.GetJobLogs(JobRequest(auth_info=self.auth_info ,job = self.job_pb))
    for job_log in data_iter:
      for log in job_log.log:
        f.write(log)
        f.flush()

  #
  # APIs for runs
  #

  def get_run_log(self, run_id):
    x = nbox_ws_v1.job.u(self.job_pb.id).runs.u(run_id).logs()
    return x

  def get_runs(self, page = -1, limit = 10) -> List[Dict]:
    """
    Get runs for this job

    Args:
      page (int, optional): Page number. Defaults to -1.
      sort (str, optional): Sort by. Defaults to "s_no".
      limit (int, optional): Number of runs to return. Defaults to 10.

    Returns:
      List[Dict]: List of runs
    """
    return self.run_stub(limit = limit, page = page)["runs_list"]

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
      return data

    runs = self.get_runs(page = page, limit = limit) # time should be reverse
    runs = sorted(runs, key = lambda x: x[sort])
    data = _display_runs(runs)
    if page == -1:
      page = 1

    done = True
    if len(data) == limit:
      y = input(f">> Print {limit} more runs? (y/n): ")
      done = y != "y"

    while not done:
      runs = self.get_runs(page = page + 1, limit = limit)
      runs = sorted(runs, key = lambda x: x[sort])
      data = _display_runs(runs)
      if len(data) < limit:
        break
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
    out = self.get_runs(page = _page, limit = n)
    all_items.extend(out)

    while len(all_items) < n:
      _page += 1
      out = self.get_runs(page = _page, limit = n)
      if not len(out):
        break
      all_items.extend(out)

    if n == 1:
      return all_items[0]
    return all_items
