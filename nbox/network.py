#!/usr/bin/env 

# this file has methods for netorking related things
import json
import requests
from time import sleep

def ocd(onnx_model_path, access_token, verbose = False):
  if verbose:
    print("=" * 30 + " NBX-OCD " + "=" * 30)

  # get the one-time-url from webserver
  r = requests.post(
    url = "https://nimblebox.ai/api/model/get_upload_url",
    params = {
      "file_size": 98,
      "file_type": filepath.split(".")[-1],
      "model_name": model_name,
    },
    header = {
      "Authorization": f"Bearer {access_token}"
    }
  )
  r.raise_for_status()
  out = r.json()

  model_id = out["fields"]["x-amz-meta-model_id"]

  # upload the file to a S3
  url = out["url"]
  files = {"file": (open(filepath, "rb"), )}
  header = {"Content-type": "multipart/form-data"}
  form = out["fields"]
  r = requests.post(url, data = form, files = files, headers = header)
  r.raise_for_status()
  
  # checking if file is successfully uploaded on S3 and tell webserver
  # whether upload is completed or not because client tells
  if r.status_code == 200:
    requests.post(
      url = "https://nimblebox.ai/api/model/update_model_status",
      json = {
        "upload": True,
        "model_id": model_id
      },
      header = {
        "Authorization": f"Bearer {access_token}"
      }
    )
  else:
    requests.post(
      url = "https://nimblebox.ai/api/model/update_model_status",
      json = {
        "upload": False,
        "model_id": model_id
      },
      header = {
        "Authorization": f"Bearer {access_token}"
      }
    )

  out = r.json()

  # polling
  # These status would already be completed
  # "upload.in-progress", "upload.success"
  endpoint = None
  while True:
    sleep(5)
    r = requests.get(
        url = "https://nimblebox.ai/api/model/get_model_history",
        params = {
            "model_id": model_id
        }
    )
    r.raise_for_status()
    statuses = r.json()["data"]
    
    if len(statuses):
      curr_st = statuses[-1]
      if "failed" in curr_st["status"]:
        msg = curr_st["status"]; msg_time = curr_st["time"]
        break

      # steps = ["upload.in-progress", 
      # "upload.success", 
      # "upload.failed", 
      # "conversion.in-progress", 
      # "conversion.success", 
      # "conversion.failed", 
      # "deployment.in-progress", 
      # "deployment.success", 
      # "deployment.failed",]

      if curr_st["status"] == "deployment.success":
        endpoint = curr_st["endpoint"]
        break
  
  return endpoint


ocd(access_token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9")