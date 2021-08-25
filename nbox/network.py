# this file has methods for netorking related things

import os
import requests
from time import sleep, time
from rich.console import Console
from pprint import pprint as peepee

import torch
from nbox import utils

URL = "https://shubham.test-2.nimblebox.ai"

class T:
  clk = "deep_sky_blue1"     # timer
  st = "bold dark_cyan"      # status + print
  fail = "bold red"          # fail
  inp = "bold yellow"        # in-progress
  nbx = "bold bright_black"  # text with NBX at top and bottom
  rule = "dark_cyan"         # ruler at top and bottom
  spinner = "weather"        # status theme


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
    print()
    console = Console()
    console.rule(f"[{T.nbx}]NBX-OCD[/{T.nbx}]", style = T.rule)
    st = time()

    # get the access tokens
    with console.status("", spinner = T.spinner) as status:
        status.update(f"[{T.st}]Getting access tokens ...[/{T.st}]")
        access_token = os.getenv("NBX_ACCESS_TOKEN", None)
        if not access_token:
            if not (username or password):
                raise ValueError("No access token found and username and password not provided")
            r = requests.post(
                url = f"{URL}/api/login",
                json = {"username": username, "password": password},
                verify=False,
            )
            try:
                r.raise_for_status()
            except:
                peepee(r.content)
            access_packet = r.json()
            access_token = access_packet.get("access_token", None)
            if access_token is None:
                raise ValueError(f"Authentication Failed: {access_token['error']}")
    console.print(f"[[{T.clk}]{utils.get_time_str(st)}[/{T.clk}]] Access token obtained")

    # convert the model
    onnx_model_path = os.path.abspath(utils.join(cache_dir, "sample.onnx"))
    if not os.path.exists(onnx_model_path):
        with console.status("", spinner = T.spinner) as status:
            status.update(f"[{T.st}]Getting access tokens ...[/{T.st}]")
            torch.onnx.export(
                model,
                args=args,
                f=onnx_model_path,
                input_names=input_names,
                verbose = verbose,
                output_names=output_names,
                
                use_external_data_format=False, # RuntimeError: Exporting model exceed maximum protobuf size of 2GB
                export_params=True,             # store the trained parameter weights inside the model file
                opset_version=12,               # the ONNX version to export the model to
                do_constant_folding=True,       # whether to execute constant folding for optimization

                dynamic_axes = dynamic_axes
            )
        console.print(f"[[{T.clk}]{utils.get_time_str(st)}[/{T.clk}]] torch -> ONNX conversion done")

    # get the one-time-url from webserver
    model_name = model_name if model_name is not None else f"{utils.get_random_name().replace('-', '_')}_{utils.hash_(model_key)}"
    console.print(f"[[{T.clk}]{utils.get_time_str(st)}[/{T.clk}]] model_name: {model_name}")
    with console.status("", spinner = T.spinner) as status:
        status.update(f"[{T.st}]Getting upload URL ...[/{T.st}]")
        r = requests.get(
            url = f"{URL}/api/model/get_upload_url",
            params = {
                "file_size": os.stat(onnx_model_path).st_size // (1024 ** 3), # because in MB
                "file_type": onnx_model_path.split(".")[-1],
                "model_name": model_name,
            },
            headers = {"Authorization": f"Bearer {access_token}"},
            verify=False,
        )
        try:
            r.raise_for_status()
        except:
            peepee(r.content)
        out = r.json()
    console.print(f"[[{T.clk}]{utils.get_time_str(st)}[/{T.clk}]] Upload URL obtained")
    model_id = out["fields"]["x-amz-meta-model_id"]
    console.print(f"[[{T.clk}]{utils.get_time_str(st)}[/{T.clk}]] model_id: {model_id}")

    # upload the file to a S3
    with console.status("", spinner = T.spinner) as status:
        status.update(f"[{T.st}]Uploading model to S3 ...[/{T.st}]")
        r = requests.post(
            out["url"],
            data = out["fields"],
            files = {"file": (out["fields"]["key"], open(onnx_model_path, "rb"))}
        )
    console.print(f"[[{T.clk}]{utils.get_time_str(st)}[/{T.clk}]] Upload to S3 complete")
    
    # checking if file is successfully uploaded on S3 and tell webserver
    # whether upload is completed or not because client tells
    with console.status("", spinner = T.spinner) as status:
        status.update(f"[{T.st}]Verifying upload ...[/{T.st}]")
        if r.status_code == 204:
            requests.post(
                url = f"{URL}/api/model/update_model_status",
                json = {"upload": True, "model_id": model_id},
                headers = {"Authorization": f"Bearer {access_token}"},
                verify = False
            )
        else:
            requests.post(
                url = f"{URL}/api/model/update_model_status",
                json = {"upload": False, "model_id": model_id},
                headers = {"Authorization": f"Bearer {access_token}"},
                verify = False
            )

    # polling
    # "upload.in-progress", "upload.success" would already be completed
    endpoint = None
    console.print(f"[[{T.clk}]{utils.get_time_str(st)}[/{T.clk}]] Start Polling ...")
    _stat_done = 0
    with console.status("", spinner = T.spinner) as status:
        while True:
            sleep_seconds = 5
            for i in range(sleep_seconds):
                status.update(f"[[{T.clk}]{utils.get_time_str(st)}[/{T.clk}]] [{T.st}]Sleeping for {sleep_seconds-i}s ...[/{T.st}]")
                sleep(1)
            status.update(f"[[{T.clk}]{utils.get_time_str(st)}[/{T.clk}]] [{T.st}]Getting updates ...[/{T.st}]")
            r = requests.get(
                url = f"{URL}/api/model/get_model_history",
                params = {"model_id": model_id},
                headers = {"Authorization": f"Bearer {access_token}"},
                verify = False
            )
            try:
                r.raise_for_status()
            except:
                peepee(r.content)

            statuses = r.json()["model_history"]

            if len(statuses):
                curr_st = statuses[-1]
                if "failed" in curr_st["status"]:
                    msg = curr_st["status"]
                    console.print(f"[[{T.clk}]{utils.get_time_str(st)}[/{T.clk}]] Status: [{T.fail}]fail[/{T.fail}] with message:")
                    console.print(f"[[{T.clk}]{utils.get_time_str(st)}[/{T.clk}]]         {msg}")
                    break
                
                if _stat_done < len(statuses):
                    status.update(f"[[{T.clk}]{utils.get_time_str(st)}[/{T.clk}]] Status: [{T.st}]{curr_st['status']}[/{T.st}]")
                    _stat_done = len(statuses)

                if curr_st["status"] == "deployment.success":
                    endpoint = curr_st["model_data"]["api_url"]
                    console.print(f"[[{T.clk}]{utils.get_time_str(st)}[/{T.clk}]] [dark_cyan]Deployment successful at URL:[/dark_cyan]")
                    console.print(f"[[{T.clk}]{utils.get_time_str(st)}[/{T.clk}]]     {endpoint}")
                    break
            
        if endpoint:
            console.rule(f"[{T.st}]NBX-OCD Success[/{T.st}]", style = T.rule)
        else:
            console.rule(f"[{T.st}]NBX-OCD Failed[/{T.st}]", style = T.rule)
    
    return endpoint
