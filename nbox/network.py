# this file has methods for netorking related things

from datetime import datetime, timedelta
import json
import os
import requests
from pprint import pprint as pp
from time import sleep

from . import utils

import logging
logger = logging.getLogger()

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
  logger.info("-" * 30 + " NBX Deploy " + "-" * 30)
  logger.info(f"Deploying on URL: {URL}")
  deployment_type = nbox_meta["spec"]["deployment_type"]
  deployment_id = nbox_meta["spec"]["deployment_id"]
  deployment_name = nbox_meta["spec"]["deployment_name"]
  model_name = nbox_meta["spec"]["model_name"]
  
  logger.info(f"Deployment Type: '{deployment_type}', Deployment ID: '{deployment_id}'")

  if not deployment_id and not deployment_name:
    logger.info("Deployment ID not passed will create a new deployment with name >>")
    deployment_name = utils.get_random_name().replace("-", "_")

  logger.info(
    f"Deployment Name: '{deployment_name}', Model Name: '{model_name}', Model Path: '{export_model_path}', file_size: {file_size} MBs"
  )
  logger.info("Getting bucket URL")

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
  logger.info(f"model_id: {model_id}")
  logger.info(f"deployment_id: {deployment_id}")

  # upload the file to a S3 -> don't raise for status here
  logger.info("Uploading model to S3 ...")
  r = requests.post(url=out["url"], data=out["fields"], files={"file": (out["fields"]["key"], open(export_model_path, "rb"))})

  # checking if file is successfully uploaded on S3 and tell webserver
  # whether upload is completed or not because client tells
  logger.info("Verifying upload ...")
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
  logger.info(f"Check your deployment at {URL}/oneclick")
  if not wait_for_deployment:
    logger.info("NBX Deploy")
    return endpoint, access_key

  logger.info("Start Polling ...")
  while True:
    total_retries += 1

    # don't keep polling for very long, kill after sometime
    if total_retries > 50 and not wait_for_deployment:
      logger.info(f"Stopping polling, please check status at: {URL}/oneclick")
      break

    sleep(5)

    # get the status update
    logger.info(f"Getting updates ...")
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
      logger.info(f"Status: {curr_st}")
      _stat_done.append(curr_st)

    if curr_st == "deployment.success":
      # if we do not have api key then query web server for it
      if access_key is None:
        endpoint = updates["model_data"]["api_url"]

        if endpoint is None:
          if wait_for_deployment:
            continue
          logger.info("Deployment in progress ...")
          logger.info(f"Endpoint to be setup, please check status at: {URL}/oneclick")
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
        logger.info(f"nbx-key: {access_key}")
      except:
        pp(r.content.decode("utf-8"))
        raise ValueError(f"Failed to get access_key, please check status at: {URL}/oneclick")

      # keep hitting /metadata and see if model is ready or not
      r = requests.get(url=f"{endpoint}/metadata", headers={"NBX-KEY": access_key, "Authorization": f"Bearer {access_token}"})
      if r.status_code == 200:
        logger.info(f"Model is ready")
        break

    # actual break condition happens here: bug in webserver where it does not return ready
    # curr_st == "ready"
    if access_key != None or "failed" in curr_st:
      break

  logger.info("Process Complete")
  logger.info("NBX Deploy")
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
    hour: int,
    minute: int,
    days: list = [],
    months: list = [],
    starts: datetime = None,
    ends: datetime = None
  ):
    # check values out of range

    self.hour = hour
    self.minute = minute

    diff = set(days) - set(self._days.keys())
    if diff != set():
      raise ValueError(f"Invalid days: {diff}")
    self.days = ",".join([self._days[d] for d in days]) if days else "*"

    diff = set(months) - set(self._months.keys())
    if diff != set():
      raise ValueError(f"Invalid months: {diff}")
    self.months = ",".join([self._months[m] for m in months]) if months else "*"

    self.starts = starts.isoformat()
    self.ends = ends.isoformat()

  @property
  def cron(self):
    return f"{self.minute} {self.hour} * {self.months} {self.days}"

  def get_dict(self):
    return {
      "cron": self.cron,
      "starts": self.starts,
      "ends": self.ends,
    }

  def __repr__(self):
    return str(self.get_dict())


def deploy_job(
  zip_path: str,
  schedule: Cron,
  data: dict
):
  from nbox.auth import secret # it can refresh so add it in the method

  access_token = secret.get("access_token")
  URL = secret.get("nbx_url")
  file_size = os.stat(zip_path).st_size // (1024 ** 2) # in MBs

  # intialise the console logger
  logger.info("-" * 30 + " NBX Jobs " + "-" * 30)
  logger.info(f"Deploying on URL: {URL}")

  # gRPC baby
  from .hyperloop.nbox_ws_pb2 import UploadCodeRequest, CreateJobRequest
  from .hyperloop.job_pb2 import Job, NBXAuthInfo
  from .hyperloop.dag_pb2 import DAG

  from google.protobuf.timestamp_pb2 import Timestamp
  from google.protobuf.json_format import MessageToJson, ParseDict

  try:
    job = utils.nbx_stub(
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
            cron = None
          ),
        ),
        auth = NBXAuthInfo(
          username=secret.get("username"),
          workspace=None
        ),
      ),
      metadata = [
        ("authorization", f"{access_token}"),
      ]
    )
  except Exception as e:
    logger.info(f"Failed to deploy job: {e}")
    return
  
  out = MessageToJson(job.code)
  s3_url = out["s3_url"]
  s3_meta = out["s3_meta"]

  job_id = s3_meta["x-amz-meta-job_id"]
  jobs_deployment_id = s3_meta["x-amz-meta-jobs_deployment_id"]
  logger.info(f"job_id: {job_id}")
  logger.info(f"jobs_deployment_id: {jobs_deployment_id}")

  # upload the file to a S3 -> don't raise for status here
  logger.info("Uploading model to S3 ...")
  r = requests.post(url=s3_url, data=s3_meta, files={"file": (s3_meta["key"], open(zip_path, "rb"))})

  # Once the file is loaded create a new job
  logger.info("Creating new job ...")
  try:
    job = utils.nbx_stub(
      CreateJobRequest(
        job = job
      )
    )
  except Exception as e:
    logger.info(f"Failed to create job: {e}")
    return

  return job
