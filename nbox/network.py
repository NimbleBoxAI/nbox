# this file has methods for netorking related things

import os
import requests
from time import sleep, time
from rich.console import Console
from pprint import pprint as peepee

import torch
from nbox import utils

URL = ""


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
    st = time()
    console = Console()
    console.rule("[bold grey0]NBX-OCD[/bold grey0]", style = "dark_cyan")

    # get the access tokens
    with console.status("", spinner = "moon") as status:
        status.update("[bold dark_cyan]Getting access tokens ...[/bold dark_cyan]")
        access_token = os.getenv("NBX_ACCESS_TOKEN", None)
        if not access_token:
            if not (username or password):
                raise ValueError("No access token found and username and password not provided")
            r = requests.post(
                url = f"{URL}/api/login",
                json = {"username": username, "password": password},
                verify=False,
            )
            r.raise_for_status()
            access_packet = r.json()
            access_token = access_packet.get("access_token", None)
            if access_token is None:
                raise ValueError(f"Authentication Failed: {access_token['error']}")
    console.print(f"[[deep_sky_blue1]{utils.get_time_str(st)}[/deep_sky_blue1]] Access token obtained")

    # convert the model
    onnx_model_path = os.path.abspath(utils.join(cache_dir, "sample.onnx"))
    if not os.path.exists(onnx_model_path):
        with console.status("", spinner = "moon") as status:
            status.update("[bold dark_cyan]Converting torch model to ONNX ...[/bold dark_cyan]")
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
        console.print(f"[[deep_sky_blue1]{utils.get_time_str(st)}[/deep_sky_blue1]] torch -> ONNX conversion done")

    # get the one-time-url from webserver
    model_name = model_name if model_name is not None else f"{utils.get_random_name().replace('-', '_')}_{utils.hash_(model_key)}"
    console.print(f"[[deep_sky_blue1]{utils.get_time_str(st)}[/deep_sky_blue1]] model_name: {model_name}")
    with console.status("", spinner = "moon") as status:
        status.update("[bold dark_cyan]Getting upload URL ...[/bold dark_cyan]")
        r = requests.get(
            url = f"{URL}/api/model/get_upload_url",
            params = {
                "file_size": os.stat(onnx_model_path).st_size // (1024 ** 3), # because in MB
                "file_type": "."+onnx_model_path.split(".")[-1],
                "model_name": model_name,
            },
            headers = {"Authorization": f"Bearer {access_token}"},
            verify=False,
        )
        r.raise_for_status()
        out = r.json()
    console.print(f"[[deep_sky_blue1]{utils.get_time_str(st)}[/deep_sky_blue1]] Upload URL obtained")
    model_id = out["fields"]["x-amz-meta-model_id"]
    console.print(f"[[deep_sky_blue1]{utils.get_time_str(st)}[/deep_sky_blue1]] model_id: {model_id}")

    # upload the file to a S3
    with console.status("", spinner = "moon") as status:
        status.update("[bold dark_cyan]Uploading model to S3 ...[/bold dark_cyan]")
        r = requests.post(
            out["url"],
            data = out["fields"],
            files = {"file": (out["fields"]["key"], open(onnx_model_path, "rb"))}
        )
    console.print(f"[[deep_sky_blue1]{utils.get_time_str(st)}[/deep_sky_blue1]] Upload to S3 complete")
    
    # checking if file is successfully uploaded on S3 and tell webserver
    # whether upload is completed or not because client tells
    with console.status("", spinner = "moon") as status:
        status.update("[bold dark_cyan]Verifying upload ...[/bold dark_cyan]")
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
    console.print(f"[[deep_sky_blue1]{utils.get_time_str(st)}[/deep_sky_blue1]] Start Polling ...")
    _stat_done = 0
    with console.status("", spinner = "moon") as status:
        while True:
            for i in range(2):
                status.update(f"[[deep_sky_blue1]{utils.get_time_str(st)}[/deep_sky_blue1]] [bold dark_cyan]Sleeping for {10-i}s ...[/bold dark_cyan]")
                sleep(1)
            r = requests.get(
                url = f"{URL}/api/model/get_model_history",
                params = {"model_id": model_id},
                headers = {"Authorization": f"Bearer {access_token}"},
                verify = False
            )
            r.raise_for_status()
            statuses = r.json()["data"]

            if len(statuses):
                curr_st = statuses[-1]
                if "failed" in curr_st["status"]:
                    msg = curr_st["status"]
                    console.print(f"[[deep_sky_blue1]{utils.get_time_str(st)}[/deep_sky_blue1]] Status: [code red]fail[/code red] with message:")
                    console.print(f"[[deep_sky_blue1]{utils.get_time_str(st)}[/deep_sky_blue1]]         {msg}")
                    break
                
                if _stat_done < len(statuses):
                    status.update(f"[[deep_sky_blue1]{utils.get_time_str(st)}[/deep_sky_blue1]] Status: [code yellow]{curr_st['status']}[/code yellow]")
                    _stat_done = len(statuses)

                if curr_st["status"] == "deployment.success":
                    endpoint = curr_st["endpoint"]
                    console.print(f"[[deep_sky_blue1]{utils.get_time_str(st)}[/deep_sky_blue1]] [dark_cyan]Deployment successful at URL:[/dark_cyan]")
                    console.print(f"[[deep_sky_blue1]{utils.get_time_str(st)}[/deep_sky_blue1]]     {endpoint}")
                    break
            
        if endpoint:
            console.rule("[bold dark_cyan]NBX-OCD Success[/bold dark_cyan]", style = "dark_cyan")
        else:
            console.rule("[bold red]NBX-OCD Failed[/bold red]", style = "dark_cyan")
    
    return endpoint
