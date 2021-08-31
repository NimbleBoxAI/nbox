# this file has methods for netorking related things

import os
import requests
from time import sleep
from typing import Dict, Tuple
from pprint import pprint as peepee

import torch
from nbox import utils

URL = os.getenv("NBX_OCD_URL")


def ocd(
    model_key: str,
    model: torch.nn.Module,
    args: Tuple,
    input_names: Tuple,
    output_names: Tuple,
    dynamic_axes: Dict,
    category: str,
    username: str = None,
    password: str = None,
    model_name: str = None,
    cache_dir: str = None,
):
    """One-Click-Deploy (OCD) method v0 that takes in the torch model, converts to ONNX
    and then deploys on NBX Platform. Avoid using this function manually and use
    `model.deploy()` instead

    Args:
        model_key (str): model_key from NBX model registry
        model (torch.nn.Module): model to be deployed
        args (Tuple): input tensor to the model for ONNX export
        input_names (Tuple): input tensor names to the model for ONNX export
        output_names (Tuple): output tensor names to the model for ONNX export
        dynamic_axes (Dict): dictionary with input_name and dynamic axes shape
        category (str): model category
        username (str, optional): your username, ignore if on NBX platform. Defaults to None.
        password (str, optional): your password, ignore if on NBX platform. Defaults to None.
        model_name (str, optional): custom model name for this model. Defaults to None.
        cache_dir (str, optional): Custom caching directory. Defaults to None.

    Raises:
        ValueError

    Returns:
        (str, None): if deployment is successful then push then return the URL endpoint else return None
    """
    print()
    console = utils.OCDConsole()
    console.rule()

    cache_dir = "/tmp" if cache_dir is None else cache_dir

    # get the access tokens
    console.start("Getting access tokens ...")
    access_token = os.getenv("NBX_ACCESS_TOKEN", None)
    if not access_token:
        if not (username or password):
            raise ValueError("No access token found and username and password not provided")
        r = requests.post(
            url=f"{URL}/api/login",
            json={"username": username, "password": password},
            verify=False,
        )
        try:
            r.raise_for_status()
        except:
            raise ValueError(f"Authentication Failed: {r.content.decode('utf-8')}")
        access_packet = r.json()
        access_token = access_packet.get("access_token", None)
        os.environ["NBX_ACCESS_TOKEN"] = access_token
    console.stop("Access token obtained")

    # convert the model
    _m_hash = utils.hash_(model_key)
    model_name = model_name if model_name is not None else f"{utils.get_random_name()}-{_m_hash[:4]}".replace("-", "_")
    console(f"model_name: {model_name}")
    onnx_model_path = os.path.abspath(utils.join(cache_dir, f"{_m_hash}.onnx"))
    if not os.path.exists(onnx_model_path):
        console.start("Converting torch -> ONNX")
        torch.onnx.export(
            model,
            args=args,
            f=onnx_model_path,
            input_names=input_names,
            verbose=False,
            output_names=output_names,
            use_external_data_format=False,  # RuntimeError: Exporting model exceed maximum protobuf size of 2GB
            export_params=True,  # store the trained parameter weights inside the model file
            opset_version=12,  # the ONNX version to export the model to
            do_constant_folding=True,  # whether to execute constant folding for optimization
            dynamic_axes=dynamic_axes,
        )
        console.stop("torch -> ONNX conversion done")

    # get the one-time-url from webserver
    console.start("Getting upload URL ...")

    # https://docs.openvinotoolkit.org/latest/openvino_docs_MO_DG_prepare_model_convert_model_Converting_Model_General.html
    input_ = ",".join(input_names)
    input_shape = ",".join([str(list(x.shape)).replace(" ", "") for x in args])
    convert_args = f"--data_type=FP32 --input_shape={input_shape} --input={input_}"

    if category == "image":
        # mean and scale have to be defined for every single input
        # these values are calcaulted from uint8 -> [-1,1] -> ImageNet scaling -> uint8
        mean_values = ",".join([f"{name}[182,178,172]" for name in input_names])
        scale_values = ",".join([f"{name}[28,27,27]" for name in input_names])
        convert_args += f" --mean_values={mean_values} --scale_values={scale_values}"

    file_size = os.stat(onnx_model_path).st_size // (1024 ** 2)  # in MBs
    console._log("convert_args:", convert_args)
    console._log("file_size:", file_size)

    r = requests.get(
        url=f"{URL}/api/model/get_upload_url",
        params={
            "file_size": file_size,  # because in MB
            "file_type": onnx_model_path.split(".")[-1],
            "model_name": model_name,
            "convert_args": convert_args,
        },
        headers={"Authorization": f"Bearer {access_token}"},
        verify=False,
    )
    try:
        r.raise_for_status()
    except:
        peepee(r.content)
    out = r.json()
    console.stop("S3 Upload URL obtained")
    model_id = out["fields"]["x-amz-meta-model_id"]
    console("model_id:", model_id)

    # upload the file to a S3
    console.start("Uploading model to S3 ...")
    r = requests.post(url=out["url"], data=out["fields"], files={"file": (out["fields"]["key"], open(onnx_model_path, "rb"))})
    # cannot raise_for_status
    console.stop(f"Upload to S3 complete")

    # checking if file is successfully uploaded on S3 and tell webserver
    # whether upload is completed or not because client tells
    console.start("Verifying upload ...")
    requests.post(
        url=f"{URL}/api/model/update_model_status",
        json={"upload": True if r.status_code == 204 else False, "model_id": model_id},
        headers={"Authorization": f"Bearer {access_token}"},
        verify=False,
    )
    console.stop("Webserver informed")

    # polling
    endpoint = None
    _stat_done = []  # status calls performed
    sleep_seconds = 3  # sleep up a little
    model_data_access_key = None  # this key is used for calling the model
    console.start("Start Polling ...")
    while True:
        for i in range(sleep_seconds):
            console(f"Sleeping for {sleep_seconds-i}s ...")
            sleep(1)

        # get the status update
        console(f"Getting updates ...")
        r = requests.get(
            url=f"{URL}/api/model/get_model_history",
            params={"model_id": model_id},
            headers={"Authorization": f"Bearer {access_token}"},
            verify=False,
        )
        try:
            r.raise_for_status()
        except:
            peepee(r.content)

        updates = r.json()
        statuses = updates["model_history"]
        if len(statuses) != len(_stat_done):
            for _st in statuses:
                curr_st = _st["status"]
                if curr_st in _stat_done:
                    continue

                # only when this is a new status
                if "failed" in curr_st:
                    console._log(f"Status: [{console.T.fail}]{curr_st}")
                elif "in-progress" in curr_st:
                    console._log(f"Status: [{console.T.inp}]{curr_st}")
                else:
                    console._log(f"Status: [{console.T.st}]{curr_st}")
                _stat_done.append(curr_st)

        # this means the deployment is done
        if statuses[-1]["status"] == "deployment.success":

            # if we do not have api key then query web server for it
            if model_data_access_key is None:
                endpoint = updates["model_data"]["api_url"]
                console._log(f"[{console.T.st}]Deployment successful at URL:\n\t{endpoint}")

                r = requests.get(
                    url=f"{URL}/get_model_access_key", headers={"Authorization": f"Bearer {access_token}"}, json={"model_id": model_id}
                )
                try:
                    r.raise_for_status()
                    model_data_access_key = r.json()["model_data_access_key"]
                    console._log(f"nbx-key: {model_data_access_key}")
                except:
                    raise ValueError(f"Failed to get model_data_access_key from /get_model_access_key")

            # keep hitting /metadata and see if model is ready or not
            r = requests.get(
                url=f"{endpoint}/metadata",
                headers={"NBX-KEY": model_data_access_key, "Authorization": f"Bearer {access_token}"},
                verify=False,
            )
            if r.status_code == 200:
                console._log(f"Model is ready")
                break

        # if failed exit
        elif "failed" in statuses[-1]["status"]:
            break

    console.stop("Process Complete")
    console.rule()
    return endpoint, model_data_access_key
