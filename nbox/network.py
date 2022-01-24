# this file has methods for netorking related things

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
        deployment_type (str, optional): type of deployment strategy
        nbox_meta (dict, optional): metadata for the nbox.Model() object being deployed
        wait_for_deployment (bool, optional): if true, acts like a blocking call (sync vs async)
        convert_args (str, optional): if deployment type == "ovms2" can pass extra arguments to MO
        deployment_id (str, optional): ``deployment_id`` to put this model under, if you do not pass this
            it will automatically create a new deployment check `platform <https://nimblebox.ai/oneclick>`_
            for more info or check the logs.
        deployment_name (str, optional): if ``deployment_id`` is not given and you want to create a new
            deployment group (ie. webserver will create a new ``deployment_id``) you can tell what name you
            want, be default it will create a random name.

    Returns:
        endpoint (str, None): if ``wait_for_deployment == True``, returns the URL endpoint of the deployed
            model
        access_key(str, None): if ``wait_for_deployment == True``, returns the data access key of
            the deployed model
    """
    from nbox.auth import secret  # it can refresh so add it in the method

    # pp(nbox_meta)

    access_token = secret.get("access_token")
    URL = secret.get("nbx_url")
    file_size = os.stat(export_model_path).st_size // (1024 ** 2)  # in MBs

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
            "file_size": file_size,  # because in MB
            "file_type": "nbox",
            "model_name": model_name,
            "convert_args": nbox_meta["spec"]["convert_args"],
            "nbox_meta": json.dumps(nbox_meta),  # annoying, but otherwise only the first key would be sent
            "deployment_type": deployment_type,  # "nbox" or "ovms2"
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
    _stat_done = []  # status calls performed
    total_retries = 0  # number of hits it took
    access_key = None  # this key is used for calling the model
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
            #     curr_st.split(".")[-1]
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

                logger.info(f"Deployment successful at URL: {endpoint}")

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


def deploy_job(
    zip_path: str,
    schedule_meta: dict,
):
    from nbox.auth import secret  # it can refresh so add it in the method

    access_token = secret.get("access_token")
    URL = secret.get("nbx_url")
    file_size = os.stat(zip_path).st_size // (1024 ** 2)  # in MBs

    # intialise the console logger
    logger.info("-" * 30 + " NBX Deploy " + "-" * 30)
    logger.info(f"Deploying on URL: {URL}")

    # POST not GET !change vs. model
    r = requests.post(
        url=f"{URL}/api/jobs/get_upload_url",
        headers={"Authorization": f"Bearer {access_token}"},
        json = {
            "file_size": file_size,  # because in MB
            "file_type": "zip",
            "job_name": schedule_meta["job_name"],
            "schedule_meta": schedule_meta,
        }
    )
    try:
        r.raise_for_status()
    except:
        raise ValueError(f"Could not fetch upload URL: {r.content.decode('utf-8')}")

    out = r.json()
    job_id = out["fields"]["x-amz-meta-job_id"]
    jobs_deployment_id = out["fields"]["x-amz-meta-jobs_deployment_id"]
    logger.info(f"job_id: {job_id}")
    logger.info(f"jobs_deployment_id: {jobs_deployment_id}")

    # upload the file to a S3 -> don't raise for status here
    logger.info("Uploading model to S3 ...")
    r = requests.post(url=out["url"], data=out["fields"], files={"file": (out["fields"]["key"], open(zip_path, "rb"))})

    # checking if file is successfully uploaded on S3 and tell webserver whether upload is completed or not because client tells
    logger.info("Verifying upload ...")
    r = requests.post(
        url=f"{URL}/api/jobs/update_model_status",
        json={"upload": True if r.status_code == 204 else False, "job_id": job_id, "jobs_deployment_id": jobs_deployment_id},
        headers={"Authorization": f"Bearer {access_token}"},
    )
    try:
        r.raise_for_status()
    except:
        raise ValueError(f"Could not update model status: {r.content.decode('utf-8')}")


