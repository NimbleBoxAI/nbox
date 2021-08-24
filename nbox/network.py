#!/usr/bin/env python3

# this file has methods for netorking related things
import os
import requests
from time import sleep, time

from pprint import pprint as peepee

from nbox import utils

import torch

def ocd(
  model,
  model_key,
  args,
  input_names,
  output_names,
  dynamic_axes,
  username,
  password,
  cache_dir,
  model_name,
  verbose = False
):
  if verbose:
    print("=" * 30 + " NBX-OCD " + "=" * 30)

  # get the access tokens
  access_token = os.getenv("NBX_ACCESS_TOKEN", None)
  if not access_token:
    if not (username or password):
      raise ValueError("No access token found and username and password not provided")
    r = requests.post(
      url = "{URL}/api/login",
      json = {
        "username": username,
        "password": password
      },
      verify=False,
    )
    r.raise_for_status()
    access_packet = r.json()
    access_token = access_packet.get("access_token", None)
    if access_token is None:
        raise ValueError(f"Authentication Failed: {access_token['error']}")
  print("--> access_token:", access_token)

  # convert the model
  onnx_model_path = os.path.abspath(utils.join(cache_dir, "sample.onnx"))
  print("--> filepath:", onnx_model_path)
  if not os.path.exists(onnx_model_path):
    torch.onnx.export(
      model,
      args=args,
      f=onnx_model_path,
      input_names=input_names,
      verbose = verbose,
      output_names=output_names,
      
      use_external_data_format=False, # RuntimeError: Exporting model exceed maximum protobuf size of 2GB
      export_params=True,        # store the trained parameter weights inside the model file
      opset_version=12,          # the ONNX version to export the model to
      do_constant_folding=True,  # whether to execute constant folding for optimization

      dynamic_axes = dynamic_axes
    )

  # get the one-time-url from webserver
  model_name = model_name if model_name is not None else f"{utils.get_random_name()}-{utils.hash_(model_key)}"
  print("--> Model_name:", model_name)

  r = requests.get(
    url = "{URL}/api/model/get_upload_url",
    params = {
      "file_size": os.stat(onnx_model_path).st_size // (1024 ** 3), # because in MB
      "file_type": "."+onnx_model_path.split(".")[-1],
      "model_name": model_name,
    },
    headers = {
      "Authorization": f"Bearer {access_token}"
    },
    verify=False,
  )
  r.raise_for_status()
  out = r.json()
  peepee(out)

  model_id = out["fields"]["x-amz-meta-model_id"]
  print("---> model_id:", model_id)

  # upload the file to a S3
  r = requests.post(
    out["url"],
    data = out["fields"],
    files  = {"file": (out["fields"]["key"], open(onnx_model_path, "rb"))}
  )
  peepee(r.content)
  print(r.status_code)
  
  # checking if file is successfully uploaded on S3 and tell webserver
  # whether upload is completed or not because client tells
  if r.status_code == 204:
    requests.post(
      url = "{URL}/api/model/update_model_status",
      json = {
        "upload": True,
        "model_id": model_id
      },
      headers = {
        "Authorization": f"Bearer {access_token}"
      },
      verify = False
    )
  else:
    requests.post(
      url = "{URL}/api/model/update_model_status",
      json = {
        "upload": False,
        "model_id": model_id
      },
      headers = {
        "Authorization": f"Bearer {access_token}"
      },
      verify = False
    )

  # out = r.json()
  peepee(out)

  # polling
  # These status would already be completed
  # "upload.in-progress", "upload.success"
  endpoint = None
  _st = time()
  while True:

    sleep(2)
    r = requests.get(
        url = "{URL}/api/model/get_model_history",
        params = {
            "model_id": model_id
        },
        headers = {
        "Authorization": f"Bearer {access_token}"
        },
        verify = False
    )
    # r.raise_for_status()
    statuses = r.json()["data"]

    peepee(statuses)
    
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


# ocd(
#   onnx_model_path = "/Users/yashbonde/Desktop/wrk/nbx/rnd/nbox/tests/__ignore/sample.onnx",
#   model_name = "progressive-granite-bfa8c39063e6c1af55087f1e401bf732",
#   access_token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJmcmVzaCI6ZmFsc2UsImlhdCI6MTYyOTgxNjMyMCwianRpIjoiMGQ1NmQ5YTQtOGU4My00YmVmLWFhMWEtZDIwMDQ1YjhmMzczIiwidHlwZSI6ImFjY2VzcyIsInN1YiI6eyJ1c2VyX2lkIjoxMzksInVzZXIiOiJ5YXNoX2JvbmRlXzEzOSIsImluc3RhbmNlIjotMSwiaXAiOiIwLjAuMC4wIiwiZW1haWwiOiJib25kZS55YXNoOTdAZ21haWwuY29tIn0sIm5iZiI6MTYyOTgxNjMyMH0.2AKPHoQMtZmha9N-UFAD1oYJnh0PNjykD8Qgq2h5wBs",
#   verbose = True
# )
