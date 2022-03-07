# this file has methods for netorking related things

import os
import json
import requests
from time import sleep
from pprint import pprint as pp
from datetime import datetime, timedelta

from .utils import logger
from . import utils


class NBXAPIError(Exception):
  pass


def deploy_model(
  export_model_path,
  nbox_meta,
  wait_for_deployment=False,
):
  """One-Click-Deploy method v3 that takes in a .nbox file and deploys it to the nbox server.
  Avoid using this function manually and use ``model.deploy()`` or nboxCLI instead.

  Args:
    export_model_path (str): path to the file to upload
    nbox_meta (dict, optional): metadata for the nbox.Model() object being deployed
    wait_for_deployment (bool, optional): if true, acts like a blocking call (sync vs async)

  Returns:
    endpoint (str, None): if ``wait_for_deployment == True``, returns the URL endpoint of the deployed
      model
    access_key(str, None): if ``wait_for_deployment == True``, returns the data access key of
      the deployed model
  """
  from nbox.auth import secret # it can refresh so add it in the method

  # pp(nbox_meta)

  access_token = secret.get("access_token")
  URL = secret.get("nbx_url")
  file_size = os.stat(export_model_path).st_size // (1024 ** 2) # in MBs

  # intialise the console logger
  logger.debug("-" * 30 + " NBX Deploy " + "-" * 30)
  logger.debug(f"Deploying on URL: {URL}")
  deployment_type = nbox_meta["spec"]["deployment_type"]
  deployment_id = nbox_meta["spec"]["deployment_id"]
  deployment_name = nbox_meta["spec"]["deployment_name"]
  model_name = nbox_meta["spec"]["model_name"]

  logger.debug(f"Deployment Type: '{deployment_type}', Deployment ID: '{deployment_id}'")

  if not deployment_id and not deployment_name:
    logger.debug("Deployment ID not passed will create a new deployment with name >>")
    deployment_name = utils.get_random_name().replace("-", "_")

  logger.debug(
    f"Deployment Name: '{deployment_name}', Model Name: '{model_name}', Model Path: '{export_model_path}', file_size: {file_size} MBs"
  )
  logger.debug("Getting bucket URL")

  # get bucket URL
  r = requests.get(
    url=f"{URL}/api/model/get_upload_url",
    params={
      "file_size": file_size, # because in MB
      "file_type": "nbox",
      "model_name": model_name,
      "convert_args": nbox_meta["spec"]["convert_args"],
      "nbox_meta": json.dumps(nbox_meta), # annoying, but otherwise only the first key would be sent
      "deployment_type": deployment_type, # "nbox" or "ovms2"
      "deployment_id": deployment_id,
      "deployment_name": deployment_name,
    },
    headers={"Authorization": f"Bearer {access_token}"},
  )
  try:
    r.raise_for_status()
  except:
    raise ValueError(f"Could not fetch upload URL: {r.content.decode('utf-8')}")
  out = r.json()
  model_id = out["fields"]["x-amz-meta-model_id"]
  deployment_id = out["fields"]["x-amz-meta-deployment_id"]
  logger.debug(f"model_id: {model_id}")
  logger.debug(f"deployment_id: {deployment_id}")

  # upload the file to a S3 -> don't raise for status here
  logger.debug("Uploading model to S3 ...")
  r = requests.post(url=out["url"], data=out["fields"], files={"file": (out["fields"]["key"], open(export_model_path, "rb"))})

  # checking if file is successfully uploaded on S3 and tell webserver
  # whether upload is completed or not because client tells
  logger.debug("Verifying upload ...")
  requests.post(
    url=f"{URL}/api/model/update_model_status",
    json={"upload": True if r.status_code == 204 else False, "model_id": model_id, "deployment_id": deployment_id},
    headers={"Authorization": f"Bearer {access_token}"},
  )

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
    r = requests.get(
      url=f"{URL}/api/model/get_model_history",
      params={"model_id": model_id, "deployment_id": deployment_id},
      headers={"Authorization": f"Bearer {access_token}"},
    )
    try:
      r.raise_for_status()
      updates = r.json()
    except:
      pp(r.content)
      raise NBXAPIError("This should not happen, please raise an issue at https://github.com/NimbleBoxAI/nbox/issues with above log!")

    # go over all the status updates and check if the deployment is done
    for st in updates["model_history"]:
      curr_st = st["status"]
      if curr_st in _stat_done:
        continue

      # # only when this is a new status
      # col = {"failed": console.T.fail, "in-progress": console.T.inp, "success": console.T.st, "ready": console.T.st}[
      #   curr_st.split(".")[-1]
      # ]
      logger.debug(f"Status: {curr_st}")
      _stat_done.append(curr_st)

    if curr_st == "deployment.success":
      # if we do not have api key then query web server for it
      if access_key is None:
        endpoint = updates["model_data"]["api_url"]

        if endpoint is None:
          if wait_for_deployment:
            continue
          logger.debug("Deployment in progress ...")
          logger.debug(f"Endpoint to be setup, please check status at: {URL}/oneclick")
          break

    elif curr_st == "deployment.ready":
      r = requests.get(
        url=f"{URL}/api/model/get_deployment_access_key",
        headers={"Authorization": f"Bearer {access_token}"},
        params={"deployment_id": deployment_id},
      )
      try:
        r.raise_for_status()
        access_key = r.json()["access_key"]
        logger.debug(f"nbx-key: {access_key}")
      except:
        pp(r.content.decode("utf-8"))
        raise ValueError(f"Failed to get access_key, please check status at: {URL}/oneclick")

      # keep hitting /metadata and see if model is ready or not
      r = requests.get(url=f"{endpoint}/metadata", headers={"NBX-KEY": access_key, "Authorization": f"Bearer {access_token}"})
      if r.status_code == 200:
        logger.debug(f"Model is ready")
        break

    # actual break condition happens here: bug in webserver where it does not return ready
    # curr_st == "ready"
    if access_key != None or "failed" in curr_st:
      break

  logger.debug("Process Complete")
  logger.debug("NBX Deploy")
  return endpoint, access_key

class Cron:
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
    """Scheduling is nothing but a type of data sturcture that should be able to process
    all patterns that the users can throw at this.

    Usage
    -----

    .. code-block:: python

      # 4:20 everyday
      Cron(4, 0)

      # 4:20 every friday
      Cron(4, 20, ["fri"])

      # 4:20 every friday from jan to feb
      Cron(4, 20, ["fri"], ["jan", "feb"])

      # 4:20 everyday starting in 2 days and runs for 3 days
      starts = datetime.utcnow() + timedelta(days = 2) # NOTE: that time is in UTC
      Cron(4, 20, starts = starts, ends = starts + timedelta(days = 3))

      # Every 1 hour
      Cron(1)

      # Every 69 minutes
      Cron(minute = 69)

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

    starts = starts or datetime.utcnow()
    self.starts = starts.isoformat()

    ends = ends or datetime.utcnow() + timedelta(days = 7)
    self.ends = ends.isoformat()

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

  def __repr__(self):
    return str(self.get_dict())


def deploy_job(
  zip_path: str,
  schedule: Cron,
  data: dict,
  workspace: str
):
  """Deploy an NBX-Job

  Args:
      zip_path (str): Path to the zip file
      schedule (Cron): Schedule of the job
      data (dict): Metadata generated for the job
      workspace (str): Name of the workspace this is to be deployed at

  Returns:
      [type]: [description]
  """

  # upload_job(stub)
  # listJobs(stub)
  # deleteJob(stub)
  # updateDag(stub)
  # create_job(stub)
  # getJob(stub)
  updateRun(stub)



  from nbox.auth import secret, get_stub # it can refresh so add it in the method

  access_token = secret.get("access_token")
  URL = secret.get("nbx_url")
  file_size = os.stat(zip_path).st_size // (1024 ** 2) # in MBs

  # intialise the console logger
  logger.debug("-" * 30 + " NBX Jobs " + "-" * 30)
  logger.debug(f"Deploying on URL: {URL}")

  # gRPC baby
  from .hyperloop.nbox_ws_pb2 import UploadCodeRequest, CreateJobRequest
  from .hyperloop.job_pb2 import Job, NBXAuthInfo
  from .hyperloop.dag_pb2 import DAG

  from .init import nbox_grpc_stub

  from google.protobuf.timestamp_pb2 import Timestamp
  from google.protobuf.json_format import MessageToJson

  try:
    # upload_job(stub)
    job = nbox_grpc_stub(
      UploadCodeRequest(
        job=Job(
          code=Job.Code(
            size=file_size,
            type=Job.Code.Type.NBOX
          ),
          name="test_job",
          dag=DAG(
            flowchart=data["dag"],
          ),
          schedule = Job.Schedule(
            start = Timestamp(
              seconds = int(schedule.starts.timestamp()),
              nanos = 0
            ),
            end = Timestamp(
              seconds = int(schedule.ends.timestamp()),
              nanos = 0
            ),
            cron = schedule.cron
          ),
        ),
        auth = NBXAuthInfo(
          username=secret.get("username"),
          workspace=workspace
        ),
      ),
      metadata = [
        ("authorization", f"{access_token}"),
      ]
    )
  except Exception as e:
    logger.debug(f"Failed to deploy job: {e}")
    return

  out = MessageToJson(job.code)
  s3_url = out["s3_url"]
  s3_meta = out["s3_meta"]

  job_id = s3_meta["x-amz-meta-job_id"]
  jobs_deployment_id = s3_meta["x-amz-meta-jobs_deployment_id"]
  logger.debug(f"job_id: {job_id}")
  logger.debug(f"jobs_deployment_id: {jobs_deployment_id}")

  # upload the file to a S3 -> don't raise for status here
  logger.debug("Uploading model to S3 ...")
  r = requests.post(url=s3_url, data=s3_meta, files={"file": (s3_meta["key"], open(zip_path, "rb"))})
  try:
    r.raise_for_status()
  except:
    logger.error(f"Failed to upload model: {r.content.decode('utf-8')}")
    return

  # Once the file is loaded create a new job
  logger.debug("Creating new job ...")
  
  # create_job(stub)
  try:
    job = data = open("./flowchart.json").readlines()
    flowchart = Parse("\n".join(data), Flowchart(), ignore_unknown_fields=True)
    try:
      job = stub.CreateJob(
          CreateJobRequest(
              job=Job(
                  id="jt3earah",
                  dag=DAG(flowchart=flowchart),
                  schedule=Job.Schedule(cron="*/1 * * * *"),
                  auth_info=NBXAuthInfo(workspace_id="zcxdpqlk"),
              )
          )
      )
    except grpc.RpcError as e:
        print("ERROR:", e.details())
    else:
        # print(MessageToJson(response, indent=2, preserving_proto_field_name=True))
        pass
  except Exception as e:
    logger.debug(f"Failed to create job: {e}")
    return

  return job
