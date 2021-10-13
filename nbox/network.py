# this file has methods for netorking related things

import os
import json
import requests
from pprint import pp, pprint as peepee

from nbox import utils


class NBXAPIError(Exception):
    pass


def one_click_deploy(
    export_model_path: str,
    deployment_type: str = "ovms2",
    nbox_meta: dict = {},
    model_name: str = None,
    wait_for_deployment: bool = False,
    convert_args: str = None,
):
    """One-Click-Deploy (OCD) method v0 that takes in the torch model, converts to ONNX
    and then deploys on NBX Platform. Avoid using this function manually and use
    `model.deploy()` instead

    Args:
        export_model_path (str): path to the file to upload
        deployment_type (str, optional): type of deployment strategy
        nbox_meta (dict, optional): metadata for the nbox.Model() object being deployed
        model_name (str, optional): name of the model being deployed
        wait_for_deployment (bool, optional): if true, acts like a blocking call (sync vs async)
        convert_args (str, optional): if deployment type == "ovms2" can pass extra arguments to MO

    Returns:
        (str, None): if deployment is successful then push then return the URL endpoint else return None
    """
    from nbox.user import secret  # it can refresh so add it in the method

    access_token = secret.get("access_token")
    URL = secret.get("nbx_url")
    file_size = os.stat(export_model_path).st_size // (1024 ** 2)  # in MBs

    # intialise the console logger
    console = utils.Console()
    console.rule("NBX Deploy")
    console._log("Deploying on URL:", URL)
    console._log("Deployment Type:", deployment_type)
    console._log("Model Path:", export_model_path)
    console._log("file_size:", file_size, "MBs")
    console.start("Getting bucket URL")

    # get bucket URL
    r = requests.get(
        url=f"{URL}/api/model/get_upload_url",
        params={
            "file_size": file_size,  # because in MB
            "file_type": export_model_path.split(".")[-1],
            "model_name": model_name,
            "convert_args": convert_args,
            "nbox_meta": json.dumps(nbox_meta),  # annoying, but otherwise only the first key would be sent
            "deployment_type": deployment_type,  # "nbox" or "ovms2"
        },
        headers={"Authorization": f"Bearer {access_token}"},
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
        if total_retries > 50 and not wait_for_deployment:
            console._log(f"Stopping polling, please check status at: {URL}/oneclick")
            break

        console.sleep(5)

        # get the status update
        console(f"Getting updates ...")
        r = requests.get(
            url=f"{URL}/api/model/get_model_history", params={"model_id": model_id}, headers={"Authorization": f"Bearer {access_token}"}
        )
        try:
            r.raise_for_status()
            updates = r.json()
        except:
            peepee(r.content)
            raise NBXAPIError("This should not happen, please raise an issue at https://github.com/NimbleBoxAI/nbox/issues with above log!")

        # go over all the status updates and check if the deployment is done
        for st in updates["model_history"]:
            curr_st = st["status"]
            if curr_st in _stat_done:
                continue

            # only when this is a new status
            col = {"failed": console.T.fail, "in-progress": console.T.inp, "success": console.T.st, "ready": console.T.st}[
                curr_st.split(".")[-1]
            ]
            console._log(f"Status: [{col}]{curr_st}")
            _stat_done.append(curr_st)

        if curr_st == "deployment.success":
            # if we do not have api key then query web server for it
            if model_data_access_key is None:
                endpoint = updates["model_data"]["api_url"]

                if endpoint is None:
                    if wait_for_deployment:
                        continue
                    console._log("Deployment in progress ...")
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
                    pp(r.content.decode("utf-8"))
                    raise ValueError(f"Failed to get model_data_access_key, please check status at: {URL}/oneclick")

            # keep hitting /metadata and see if model is ready or not
            r = requests.get(
                url=f"{endpoint}/metadata", headers={"NBX-KEY": model_data_access_key, "Authorization": f"Bearer {access_token}"}
            )
            if r.status_code == 200:
                console._log(f"Model is ready")
                break

        # actual break condition happens here: bug in webserver where it does not return ready
        # curr_st == "ready"
        if model_data_access_key != None or "failed" in curr_st:
            break

    secret.add_ocd(
        model_id=model_id,
        url=endpoint,
        nbox_meta=nbox_meta,
        access_key=model_data_access_key,
    )

    console.stop("Process Complete")
    console.rule("NBX Deploy")
    return endpoint, model_data_access_key
