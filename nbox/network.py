# this file has methods for netorking related things

import os
import json
import requests
from time import sleep
from typing import Dict, Tuple
from pprint import pprint as peepee

import torch

from nbox import utils
from nbox.user import secret
import nbox.framework.pytorch as frm_pytorch

URL = secret.get("nbx_url")


def one_click_deploy(
    model_key: str,
    model: torch.nn.Module,
    args: Tuple,
    outputs: Tuple,
    input_names: Tuple,
    input_shapes: Tuple,
    output_names: Tuple,
    output_shapes: Tuple,
    dynamic_axes: Dict,
    category: str,
    deployment_type: str = "ovms2",
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
        input_shapes (Tuple): input tensor shapes to the model for ONNX export
        output_names (Tuple): output tensor names to the model for ONNX export
        output_shapes (Tuple): output tensor shapes to the model for ONNX export
        dynamic_axes (Dict): dictionary with input_name and dynamic axes shape
        category (str): model category
        model_name (str, optional): custom model name for this model. Defaults to None.
        cache_dir (str, optional): Custom caching directory. Defaults to None.

    Raises:
        ValueError

    Returns:
        (str, None): if deployment is successful then push then return the URL endpoint else return None
    """
    # perform sanity checks on the input values
    assert deployment_type in ["ovms2", "nbxs"], f"Only OpenVino and Nbox-Serving is supported got: {deployment_type}"

    # intialise the console logger
    console = utils.Console()
    console.rule()
    cache_dir = "/tmp" if cache_dir is None else cache_dir

    # getting access token -> now loads directly from the secret file
    access_token = secret.get("access_token")

    # convert the model
    _m_hash = utils.hash_(model_key)
    model_name = model_name if model_name is not None else f"{utils.get_random_name()}-{_m_hash[:4]}".replace("-", "_")
    console(f"model_name: {model_name}")
    spec = {"category": category, "model_key": model_key, "name": model_name}

    export_model_path = os.path.abspath(utils.join(cache_dir, _m_hash))
    if deployment_type == "ovms2":
        export_model_path += ".onnx"
        export_fn = frm_pytorch.export_to_onnx
    elif deployment_type == "nbxs":
        export_model_path += ".torchscript"
        export_fn = frm_pytorch.export_to_torchscript

    console.start(f"Converting using: {export_fn}")
    nbox_meta = export_fn(
        model=model,
        args=args,
        outputs=outputs,
        input_shapes=input_shapes,
        output_shapes=output_shapes,
        onnx_model_path=export_model_path,
        input_names=input_names,
        dynamic_axes=dynamic_axes,
        output_names=output_names,
    )
    console.stop("Conversion Complete")
    nbox_meta = {
        "metadata": nbox_meta,
        "spec": spec,
    }
    console._log("nbox_meta:", nbox_meta)

    # get the one-time-url from webserver
    console.start("Getting upload URL ...")
    convert_args = ""
    if deployment_type == "ovms2":
        # https://docs.openvinotoolkit.org/latest/openvino_docs_MO_DG_prepare_model_convert_model_Converting_Model_General.html
        input_ = ",".join(input_names)
        input_shape = ",".join([str(list(x.shape)).replace(" ", "") for x in args])
        convert_args += f"--data_type=FP32 --input_shape={input_shape} --input={input_} "

        if category == "image":
            # mean and scale have to be defined for every single input
            # these values are calcaulted from uint8 -> [-1,1] -> ImageNet scaling -> uint8
            mean_values = ",".join([f"{name}[182,178,172]" for name in input_names])
            scale_values = ",".join([f"{name}[28,27,27]" for name in input_names])
            convert_args += f"--mean_values={mean_values} --scale_values={scale_values}"

    file_size = os.stat(export_model_path).st_size // (1024 ** 2)  # in MBs
    console._log("convert_args:", convert_args)
    console._log("file_size:", file_size)

    r = requests.get(
        url=f"{URL}/api/model/get_upload_url",
        params={
            "file_size": file_size,  # because in MB
            "file_type": export_model_path.split(".")[-1],
            "model_name": model_name,
            "convert_args": convert_args,
            "nbox_meta": json.dumps(nbox_meta),  # annoying, but otherwise only the first key would be sent
            "deployment_type": deployment_type,  # "nbxs" or "ovms2"
        },
        headers={"Authorization": f"Bearer {access_token}"},
        verify=False,
    )
    try:
        r.raise_for_status()
    except:
        raise ValueError(f"Could not fetch upload URL: {r.content.decode('utf-8')}")
    out = r.json()
    model_id = out["fields"]["x-amz-meta-model_id"]
    console.stop("S3 Upload URL obtained")
    console._log("model_id:", model_id)

    # upload the file to a S3 -> don't raise for status here
    console.start("Uploading model to S3 ...")
    r = requests.post(url=out["url"], data=out["fields"], files={"file": (out["fields"]["key"], open(export_model_path, "rb"))})
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
    total_retries = 0  # number of hits it took
    model_data_access_key = None  # this key is used for calling the model
    console._log(f"Check your deployment at {URL}/oneclick")
    console.start("Start Polling ...")
    while True:
        total_retries += 1

        # don't keep polling for very long, kill after sometime
        if total_retries > 50:
            console._log(f"Stopping polling, please check status at: {URL}/oneclick")

        for i in range(3):
            console(f"Sleeping for {3-i}s ...")
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

                if endpoint is None:
                    console._log("Deployment in proress ...")
                    console._log(f"Endpoint to be setup, please check status at: {URL}/oneclick")
                    break

                console._log(f"[{console.T.st}]Deployment successful at URL:\n\t{endpoint}")

                r = requests.get(
                    url=f"{URL}/api/model/get_model_access_key",
                    headers={"Authorization": f"Bearer {access_token}"},
                    params={"model_id": model_id},
                )
                try:
                    r.raise_for_status()
                    model_data_access_key = r.json()["model_data_access_key"]
                    console._log(f"nbx-key: {model_data_access_key}")
                except:
                    raise ValueError(f"Failed to get model_data_access_key: {r.content}")

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

    secret.add_ocd(
        model_id=model_id,
        url=endpoint,
        nbox_meta=nbox_meta,
        access_key=model_data_access_key,
    )

    console.stop("Process Complete")
    console.rule()
    return endpoint, model_data_access_key
