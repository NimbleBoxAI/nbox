"""
Network functions are gateway between NBX-Services. If you find yourself using this
you might want to reach out to us <research-at-nimblebox-dot-ai>!

But for the curious mind, many of our services work on gRPC and Protobufs. This network.py
manages the quirkyness of our backend and packs multiple steps as one function.
"""

import os
import re
import grpc
import jinja2
import fnmatch
import zipfile
import requests
from tempfile import gettempdir
from datetime import datetime, timedelta, timezone
from google.protobuf.timestamp_pb2 import Timestamp
from google.protobuf.field_mask_pb2 import FieldMask

import nbox.utils as U
from nbox.auth import secret
from nbox.utils import logger
from nbox.version import __version__
from nbox.hyperloop.dag_pb2 import DAG
from nbox.init import nbox_ws_v1, nbox_grpc_stub
from nbox.hyperloop.job_pb2 import NBXAuthInfo, Job as JobProto, Resource
from nbox.messages import rpc, write_string_to_file, get_current_timestamp
from nbox.hyperloop.nbox_ws_pb2 import UploadCodeRequest, CreateJobRequest, UpdateJobRequest

class NBXAPIError(Exception):
  pass


class Schedule:
  _days = {
    k:str(i) for i,k in enumerate(
      ["sun","mon","tue","wed","thu","fri","sat"]
    )
  }
  _months = {
    k:str(i+1) for i,k in enumerate(
      ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]
    )
  }

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

    diff = set(days) - set(self._days.keys())
    if diff != set():
      raise ValueError(f"Invalid days: {diff}")
    self.days = ",".join([self._days[d] for d in days]) if days else "*"

    diff = set(months) - set(self._months.keys())
    if diff != set():
      raise ValueError(f"Invalid months: {diff}")
    self.months = ",".join([self._months[m] for m in months]) if months else "*"

    self.starts = starts or datetime.now(timezone.utc)
    self.ends = ends or datetime.now(timezone.utc) + timedelta(days = 7)

  @property
  def cron(self):
    """Cron string"""
    if self.mode == "every":
      return f"{self.minute} {self.hour} * * *"
    return f"{self.minute} {self.hour} * {self.months} {self.days}"

  def get_dict(self):
    return {
      "cron": self.cron,
      "mode": self.mode,
      "starts": self.starts,
      "ends": self.ends,
    }

  def get_message(self) -> JobProto.Schedule:
    """Get the JobProto.Schedule object for this Schedule"""
    _starts = Timestamp(); _starts.GetCurrentTime()
    _ends = Timestamp(); _ends.FromDatetime(self.ends)
    # return JobProto.Schedule(start = _starts, end = _ends, cron = self.cron)
    return JobProto.Schedule(cron = self.cron)

  def __repr__(self):
    return str(self.get_dict())


def _get_deployment_data(id_or_name, workspace_id):
  # filter and get "id" and "name"
  stub_all_depl = nbox_ws_v1.workspace.u(workspace_id).deployments
  all_jobs = stub_all_depl()
  jobs = list(filter(lambda x: x["deployment_id"] == id_or_name or x["deployment_name"] == id_or_name, all_jobs))
  if len(jobs) == 0:
    logger.info(f"No Job found with ID or name: {id_or_name}, will create a new one")
    serving_name = id_or_name
    serving_id = None
  elif len(jobs) > 1:
    raise ValueError(f"Multiple jobs found for '{id_or_name}', try passing ID")
  else:
    logger.info(f"Found job with ID or name: {id_or_name}, will update it")
    data = jobs[0]
    serving_name = data["deployment_name"]
    serving_id = data["deployment_id"]
  return serving_id, serving_name


def deploy_serving(
  init_folder: str,
  deployment_id_or_name: str,
  workspace_id: str = None,
  resource: Resource = None,
  wait_for_deployment: bool = False,
  *,
  _unittest = False
):
  """Use the NBX-Deploy Infrastructure"""
  # check if this is a valid folder or not
  if not os.path.exists(init_folder) or not os.path.isdir(init_folder):
    raise ValueError(f"Incorrect project at path: '{init_folder}'! nbox jobs new <name>")
  
  serving_id, serving_name = _get_deployment_data(deployment_id_or_name, workspace_id)
  logger.info(f"Serving name: {serving_name}")
  logger.info(f"Serving ID: {serving_id}")
  model_name = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
  logger.debug(f"Model name: {model_name}")

  # zip init folder
  zip_path = zip_to_nbox_folder(init_folder, serving_id, workspace_id, model_name = model_name)
  _upload_serving_zip(zip_path, workspace_id, serving_id, serving_name, model_name)

def _upload_serving_zip(zip_path, workspace_id, serving_id, serving_name, model_name):
  file_size = os.stat(zip_path).st_size # serving in bytes

  # get bucket URL and upload the data
  stub_all_depl = nbox_ws_v1.workspace.u(workspace_id).deployments
  out = stub_all_depl.u(serving_id).get_upload_url(
    _method = "put",
    convert_args = "",
    deployment_meta = {},
    deployment_name = serving_name,
    deployment_type = "nbox_op", # "nbox" or "ovms2"
    file_size = str(file_size),
    file_type = "nbox",
    model_name = model_name,
    nbox_meta = {},
  )

  model_id = out["fields"]["x-amz-meta-model_id"]
  deployment_id = out["fields"]["x-amz-meta-deployment_id"]
  logger.debug(f"model_id: {model_id}")
  logger.debug(f"deployment_id: {deployment_id}")

  # upload the file to a S3 -> don't raise for status here
  logger.debug("Uploading model to S3 ...")
  r = requests.post(url=out["url"], data=out["fields"], files={"file": (out["fields"]["key"], open(zip_path, "rb"))})
  status = r.status_code == 204
  logger.debug(f"Upload status: {status}")

  # checking if file is successfully uploaded on S3 and tell webserver whether upload is completed or not
  ws_stub_model = stub_all_depl.u(deployment_id).models.u(model_id) # eager create the stub
  ws_stub_model.update(_method = "post", status = status)
  logger.debug(f"Webserver informed: {r.status_code}")

  # # write out all the commands for this deployment
  # logger.info("Run is now created, to 'trigger' programatically, use the following commands:")
  # _api = f"nbox.Operator(id = '{job_proto.id}', workspace_id='{job_proto.auth_info.workspace_id}').trigger()"
  # _cli = f"python3 -m nbox jobs --id {job_proto.id} --workspace_id {job_proto.auth_info.workspace_id} trigger"
  # _curl = f"curl -X POST {secret.get('nbx_url')}/api/v1/workspace/{job_proto.auth_info.workspace_id}/job/{job_proto.id}/trigger"
  # _webpage = f"{secret.get('nbx_url')}/workspace/{job_proto.auth_info.workspace_id}/jobs/{job_proto.id}"
  # logger.info(f" [python] - {_api}")
  # logger.info(f"    [CLI] - {_cli}")
  # logger.info(f"   [curl] - {_curl} -H 'authorization: Bearer $NBX_TOKEN' -H 'Content-Type: application/json' -d " + "'{}'")
  # logger.info(f"   [page] - {_webpage}")


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
    logger.info(f"No Job found with ID or name: {id_or_name}, will create a new one")
    job_name =  id_or_name
    job_id = None
  elif len(jobs) > 1:
    raise ValueError(f"Multiple jobs found for '{id_or_name}', try passing ID")
  else:
    logger.info(f"Found job with ID or name: {id_or_name}, will update it")
    data = jobs[0]
    job_name = data["name"]
    job_id = data["job_id"]
  
  return job_id, job_name


def deploy_job(
  init_folder: str,
  job_id_or_name: str,
  dag: DAG,
  workspace_id: str = None,
  schedule: Schedule = None,
  resource: Resource = None,
  *,
  _unittest = False
) -> None:
  """Upload code for a NBX-Job.

  Args:
    init_folder (str, optional): Name the folder to zip
    job_id_or_name (Union[str, int], optional): Name or ID of the job
    dag (DAG): DAG to upload
    workspace_id (str): Workspace ID to deploy to, if not specified, will use the personal workspace
    schedule (Schedule, optional): If ``None`` will run only once, else will schedule the job
    cache_dir (str, optional): Folder where to put the zipped file, if ``None`` will be ``tempdir``
  Returns:
    Job: Job object
  """
  # check if this is a valid folder or not
  if not os.path.exists(init_folder) or not os.path.isdir(init_folder):
    raise ValueError(f"Incorrect project at path: '{init_folder}'! nbox jobs new <name>")

  job_id, job_name = _get_job_data(job_id_or_name, workspace_id)
  logger.info(f"Job name: {job_name}")
  logger.info(f"Job ID: {job_id}")

  # intialise the console logger
  URL = secret.get("nbx_url")
  logger.debug(f"Schedule: {schedule}")
  logger.debug("-" * 30 + " NBX Jobs " + "-" * 30)
  logger.debug(f"Deploying on URL: {URL}")

  # create the proto for this Operator
  job_proto = JobProto(
    id = job_id,
    name = job_name or U.get_random_name(True).split("-")[0],
    created_at = get_current_timestamp(),
    auth_info = NBXAuthInfo(
      username = secret.get("username"),
      workspace_id = workspace_id,
    ),
    schedule = schedule.get_message() if schedule is not None else None,
    dag = dag,
    resource = Resource(
      cpu = "100m",         # 100mCPU
      memory = "200Mi",     # MiB
      disk_size = "1Gi",    # GiB
    ) if resource == None else resource,
  )
  proto_path = U.join(init_folder, "job_proto.pbtxt")
  write_string_to_file(job_proto, proto_path)

  if _unittest:
    return job_proto

  # zip the entire init folder to zip, this will be response
  zip_path = zip_to_nbox_folder(init_folder, job_id, workspace_id)
  _upload_job_zip(zip_path, job_id, job_proto)

def _upload_job_zip(zip_path, job_id, job_proto):

  # incase an old job exists, we need to update few things with the new information
  if job_id != None:
    from nbox.jobs import Job
    logger.debug("Found existing job, checking for update masks")
    old_job_proto = Job(job_proto.id, job_proto.auth_info.workspace_id).job_proto
    paths = []
    if old_job_proto.resource.SerializeToString(deterministic = True) != job_proto.resource.SerializeToString(deterministic = True):
      paths.append("resource")
    if old_job_proto.schedule.cron != job_proto.schedule.cron:
      paths.append("schedule.cron")
    logger.debug(f"Updating fields: {paths}")
    nbox_grpc_stub.UpdateJob(
      UpdateJobRequest(job = job_proto, update_mask = FieldMask(paths=paths)),
    )

  # update the JobProto with file sizes
  job_proto.code.MergeFrom(JobProto.Code(
    size = max(os.stat(zip_path).st_size / (1024 ** 2), 1), # jobs in MiB
    type = JobProto.Code.Type.ZIP,
  ))

  # UploadJobCode is responsible for uploading the code of the job
  response: JobProto = rpc(
    nbox_grpc_stub.UploadJobCode,
    UploadCodeRequest(job = job_proto, auth = job_proto.auth_info),
    f"Failed to deploy job: {job_proto.id}"
  )
  job_proto.MergeFrom(response)
  s3_url = job_proto.code.s3_url
  s3_meta = job_proto.code.s3_meta

  logger.debug("Uploading model to S3 ...")
  r = requests.post(url=s3_url, data=s3_meta, files={"file": (s3_meta["key"], open(zip_path, "rb"))})
  try:
    r.raise_for_status()
  except:
    logger.error(f"Failed to upload model: {r.content.decode('utf-8')}")
    return

  logger.info("Creating new run ...")
  try:
    job: JobProto = nbox_grpc_stub.CreateJob(CreateJobRequest(job = job_proto))
  except grpc.RpcError as e:
    if e.code() == grpc.StatusCode.ALREADY_EXISTS:
      logger.debug(f"Job {job_proto.id} already exists")
    else:
      raise e
  except Exception as e:
    logger.error(f"Failed to create job: {e}")
    return

  # write out all the commands for this job
  logger.info("Run is now created, to 'trigger' programatically, use the following commands:")
  _api = f"nbox.Job(id = '{job_proto.id}', workspace_id='{job_proto.auth_info.workspace_id}').trigger()"
  _cli = f"python3 -m nbox jobs --id {job_proto.id} --workspace_id {job_proto.auth_info.workspace_id} trigger"
  _curl = f"curl -X POST {secret.get('nbx_url')}/api/v1/workspace/{job_proto.auth_info.workspace_id}/job/{job_proto.id}/trigger"
  _webpage = f"{secret.get('nbx_url')}/workspace/{job_proto.auth_info.workspace_id}/jobs/{job_proto.id}"
  logger.info(f" [python] - {_api}")
  logger.info(f"    [CLI] - {_cli}")
  logger.info(f"   [curl] - {_curl} -H 'authorization: Bearer $NBX_TOKEN' -H 'Content-Type: application/json' -d " + "'{}'")
  logger.info(f"   [page] - {_webpage}")


def zip_to_nbox_folder(init_folder, id, workspace_id, **jinja_kwargs):
  # zip all the files folder
  all_f = U.get_files_in_folder(init_folder)

  # find a .nboxignore file and ignore items in it
  to_ignore_pat = []
  to_ignore_folder = []
  for f in all_f:
    if f.split("/")[-1] == ".nboxignore":
      with open(f, "r") as _f:
        for pat in _f:
          pat = pat.strip()
          if pat.endswith("/"):
            to_ignore_folder.append(pat)
          else:
            to_ignore_pat.append(pat)
      break

  # two different lists for convinience
  to_remove = []
  for ignore in to_ignore_pat:
    x = fnmatch.filter(all_f, ignore)
    to_remove.extend(x)
  to_remove_folder = []
  for ignore in to_ignore_folder:
    for f in all_f:
      if re.search(ignore, f):
        to_remove_folder.append(f)
  to_remove += to_remove_folder

  all_f = [x for x in all_f if x not in to_remove]
  logger.info(f"Will zip {len(all_f)} files")

  # zip all the files folder
  zip_path = U.join(gettempdir(), f"nbxjd_{id}@{workspace_id}.nbox")
  logger.info(f"Packing project to '{zip_path}'")
  with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
    abspath_init_folder = os.path.abspath(init_folder)
    for f in all_f:
      arcname = f[len(abspath_init_folder)+1:]
      logger.debug(f"Zipping {f} => {arcname}")
      zip_file.write(f, arcname = arcname)

    # get a timestamp like this: Monday W34 [UTC 12 April, 2022 - 12:00:00]
    _ct = datetime.now(timezone.utc)
    _day = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][_ct.weekday()]
    created_time = f"{_day} W{_ct.isocalendar()[1]} [ UTC {_ct.strftime('%d %b, %Y - %H:%M:%S')} ]"

    # create the exe.py file
    exe_jinja_path = U.join(U.folder(__file__), "assets", "exe.jinja")
    exe_path = U.join(gettempdir(), "exe.py")
    logger.debug(f"Writing exe to: {exe_path}")
    with open(exe_jinja_path, "r") as f, open(exe_path, "w") as f2:
      f2.write(jinja2.Template(f.read()).render({
        "created_time": created_time,
        "nbox_version": __version__,
        **jinja_kwargs
      }))
    zip_file.write(exe_path, arcname = "exe.py")

  return zip_path
