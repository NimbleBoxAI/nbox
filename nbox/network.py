"""
Network functions are gateway between NBX-Services. If you find yourself using this
you might want to reach out to us <research-at-nimblebox-dot-ai>!

But for the curious mind, many of our services work on gRPC and Protobufs. This network.py
manages the quirkyness of our backend and packs multiple steps as one function.
"""

import os
import requests
from time import sleep
from datetime import datetime, timedelta

from google.protobuf.timestamp_pb2 import Timestamp

from .hyperloop.nbox_ws_pb2 import UploadCodeRequest, CreateJobRequest, UpdateJobRequest
from .hyperloop.job_pb2 import Job as JobProto

from .init import nbox_grpc_stub
from .auth import secret
from .jobs import Job
from .messages import message_to_dict, rpc
from .framework.model_spec_pb2 import ModelSpec
from .subway import Sub30
from .utils import logger
from . import utils as U
from .hyperloop.job_pb2 import Job as JobProto


class NBXAPIError(Exception):
  pass


def deploy_serving(
  export_model_path,
  stub_all_depl: Sub30,
  model_spec: ModelSpec,
  wait_for_deployment=False,
):
  """One-Click-Deploy API to serve items on a NBX-Deploy

  Args:
    export_model_path (str): path to the file to upload
    stub_all_depl (nbox.Sub30): Subway RPC stub for ``/deployments``
    model_spec (nbox.ModelSpec): ModelSpec object
    wait_for_deployment (bool, optional): if true, acts like a blocking call (sync vs async)

  Returns:
    if ``wait_for_deployment == True`` returns ``(url, key)`` pair
  """
  logger.info(f"stub_all_depl: {stub_all_depl}")
  from nbox.auth import secret # it can refresh so add it in the method
  access_token = secret.get("access_token")
  URL = secret.get("nbx_url")

  # intialise the console logger
  logger.debug("-" * 30 + " NBX Deploy " + "-" * 30)
  logger.debug(f"Deploying on URL: {URL}")
  
  # TODO: @yashbonde figure out protobuf way of storing deployment_id strings
  deployment_type = "nbox" # model_spec.deploy.type
  deployment_id = model_spec.deploy.id
  deployment_name = model_spec.deploy.name
  model_name = model_spec.name

  logger.debug(f"Deployment Type: '{deployment_type}', Deployment ID: '{deployment_id}'")

  if not deployment_id and not deployment_name:
    logger.debug("Deployment ID not passed will create a new deployment with name >>")
    deployment_name = U.get_random_name().replace("-", "_")

  file_size = os.stat(export_model_path).st_size // (1024 ** 2) # because in MB
  logger.debug(
    f"Deployment Name: '{deployment_name}', Model Name: '{model_name}', Model Path: '{export_model_path}', file_size: {file_size} MBs"
  )
  logger.debug("Getting bucket URL")

  # get bucket URL and upload the data
  out = stub_all_depl.u(deployment_id).get_upload_url(
    _method = "put",
    convert_args = "",
    deployment_meta = {},
    deployment_name = deployment_name,
    deployment_type = deployment_type, # "nbox" or "ovms2"
    file_size = str(file_size),
    file_type = "nbox",
    model_name = model_name,
    nbox_meta = message_to_dict(model_spec) , # message_to_json(model_spec), # annoying, but otherwise only the first key would be sent
    # deployment_id = deployment_id,
  )
  model_id = out["fields"]["x-amz-meta-model_id"]
  deployment_id = out["fields"]["x-amz-meta-deployment_id"]
  logger.debug(f"model_id: {model_id}")
  logger.debug(f"deployment_id: {deployment_id}")

  # upload the file to a S3 -> don't raise for status here
  logger.debug("Uploading model to S3 ...")
  r = requests.post(url=out["url"], data=out["fields"], files={"file": (out["fields"]["key"], open(export_model_path, "rb"))})

  # checking if file is successfully uploaded on S3 and tell webserver
  # whether upload is completed or not because client tells
  ws_stub_model = stub_all_depl.u(model_spec.deploy.id).models.u(model_id) # eager create the stub
  ws_stub_model.update(_method = "post", status = r.status_code == 204)

  # polling
  endpoint = None
  _stat_done = [] # status calls performed
  total_retries = 0 # number of hits it took
  access_key = None # this key is used for calling the model
  logger.debug(f"Check your deployment at {URL}/oneclick")
  if not wait_for_deployment:
    logger.debug("NBX Deploy")
    return endpoint, access_key

  logger.debug("Start Polling ...")
  while True:
    total_retries += 1

    # don't keep polling for very long, kill after sometime
    if total_retries > 50 and not wait_for_deployment:
      logger.debug(f"Stopping polling, please check status at: {URL}/oneclick")
      break

    sleep(5)

    # get the status update
    logger.debug(f"Getting updates ...")
    updates = ws_stub_model.history()["data"]
    for st in updates["model_history"]:
      curr_st = st["status"]
      logger.debug(f"Status: {curr_st}")
      if curr_st in _stat_done:
        continue
      logger.info(f"Status: {curr_st}")
      _stat_done.append(curr_st)

    if curr_st == "deployment.success":
      # if we do not have api key then query web server for it
      if access_key is None:
        server_endpoint = updates["model_data"]["api_url"]
        if server_endpoint is None:
          if wait_for_deployment:
            continue
          logger.debug("Deployment in progress ...")
          logger.debug(f"Endpoint to be setup, please check status at: {URL}/oneclick")
          break

    elif curr_st == "deployment.ready":
      out = stub_all_depl.u(model_spec.deploy.id).get_access_key()
      access_key = out["access_key"]

      # keep hitting /metadata and see if model is ready or not
      r = requests.get(url=f"{server_endpoint}/metadata", headers={"NBX-KEY": access_key, "Authorization": f"Bearer {access_token}"})
      if r.status_code == 200:
        logger.debug(f"Model is ready")
        break

    # actual break condition happens here: bug in webserver where it does not return ready
    # curr_st == "ready"
    if access_key != None or "failed" in curr_st:
      break

  logger.debug("Process Complete")
  logger.debug("NBX Deploy")
  return server_endpoint, access_key

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
      starts = datetime.utcnow() + timedelta(days = 2) # NOTE: that time is in UTC
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
      self.minute = "*"
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

    self.starts = starts or datetime.utcnow()
    self.ends = ends or datetime.utcnow() + timedelta(days = 7)

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
    return JobProto.Schedule(start = _starts, end = _ends, cron = self.cron)

  def __repr__(self):
    return str(self.get_dict())


def deploy_job(zip_path: str, job_proto: JobProto):
  """Deploy an NBX-Job

  Args:
      zip_path (str): Path to the zip file
      schedule (Schedule): Schedule of the job
  Returns:
      nbox.Job: the job object
  """

  URL = secret.get("nbx_url")
  file_size = int(os.stat(zip_path).st_size // (1024 ** 2)) # in MBs

  # intialise the console logger
  logger.debug("-" * 30 + " NBX Jobs " + "-" * 30)
  logger.debug(f"Deploying on URL: {URL}")

  code = JobProto.Code(size = max(file_size, 1), type = JobProto.Code.Type.ZIP)
  job_proto.code.MergeFrom(code)

  response: JobProto = rpc(
    nbox_grpc_stub.UploadJobCode,
    UploadCodeRequest(job = job_proto, auth = job_proto.auth_info),
    f"Failed to deploy job: {job_proto.id}"
  )

  job_proto.MergeFrom(response)
  s3_url = job_proto.code.s3_url
  s3_meta = job_proto.code.s3_meta
  logger.debug(f"Job ID: {job_proto.id}")

  logger.debug("Uploading model to S3 ...")
  r = requests.post(url=s3_url, data=s3_meta, files={"file": (s3_meta["key"], open(zip_path, "rb"))})
  try:
    r.raise_for_status()
  except:
    logger.error(f"Failed to upload model: {r.content.decode('utf-8')}")
    return
  
  logger.info("Creating new run ...")
  rpc(nbox_grpc_stub.CreateJob, CreateJobRequest(job = job_proto), f"Failed to create job: {job_proto.id}")

  return Job(job_proto.id, job_proto.auth_info.workspace_id)
