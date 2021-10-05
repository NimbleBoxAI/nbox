import json
import os
from time import sleep

import requests
from nbox.utils import Console, join


def login(username,password,nbx_home_url="https://nimblebox.ai"):
    try:
        r = requests.post(url=f"{nbx_home_url}/api/login", json={"username": username, "password": password})
    except Exception as e:
        raise Exception(f"Could not connect to NBX. You cannot use any cloud based tool!")

    if r.status_code == 401:
        print("::" * 20 + " Invalid username/password. Please try again!")
    elif r.status_code == 200:
        folder = join(os.path.expanduser("~"), ".nbx")
        os.makedirs(folder, exist_ok=True)
        fp = join(folder, "secrets.json")
        access_packet = r.json()
        access_token = access_packet.get("access_token", None)
        with open(fp, "w") as f:
                secrets = {"version": 1}
                secrets["access_token"] = access_token
                secrets["username"] = username
                secrets["nbx_url"] = nbx_home_url
                f.write(json.dumps(secrets, indent=2))
        return 
    else:
        raise Exception(f"Unknown error: {r.status_code}")
    
def deploy(model_path,model_name,category,input_names,nbox_meta,export_model,path,deployment_type="ovms2", args=[],wait_for_deployment: bool = False,):

    if not os.path.exists(join(os.path.expanduser("~"), ".nbx")):
        return "You are not login try nbox login --help"
    
    # get the one-time-url from webserver
    console = Console()
    console.rule()
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

        console._log("convert_args:", convert_args)

    
    file_size = os.stat(model_path).st_size // (1024 ** 2)  # in MBs
    console._log("file_size:", file_size)

    secrets = None
    with open(join(os.path.expanduser("~"), ".nbx"), "r") as f:
            secrets = json.load(f)
    URL = secrets["nbx_url"]
    access_token = secrets["access_token"]
    r = requests.get(
        url=f"{URL}/api/model/get_upload_url",
        params={
            "file_size": file_size,  # because in MB
            "file_type": model_path.split(".")[-1],
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
    r = requests.post(url=out["url"], data=out["fields"], files={"file": (out["fields"]["key"], open(model_path, "rb"))})
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

        for i in range(3):
            console(f"Sleeping for {3-i}s ...")
            sleep(1)

        # get the status update
        console(f"Getting updates ...")
        r = requests.get(
            url=f"{URL}/api/model/get_model_history", params={"model_id": model_id}, headers={"Authorization": f"Bearer {access_token}"}
        )
        try:
            r.raise_for_status()
        except:
            raise ValueError("This should not happen, please raise an issue at https://github.com/NimbleBoxAI/nbox/issues with above log!")

        updates = r.json()
        statuses = updates["model_history"]
        if len(statuses) != len(_stat_done):
            for _st in statuses:
                curr_st = _st["status"]
                if curr_st in _stat_done:
                    continue

                # only when this is a new status
                col = {"failed": console.T.fail, "in-progress": console.T.inp, "success": console.T.st}[curr_st.split(".")[-1]]
                console._log(f"Status: [{col}]{curr_st}")
                _stat_done.append(curr_st)

        # this means the deployment is done
        if statuses[-1]["status"] == "deployment.success":

            # if we do not have api key then query web server for it
            if model_data_access_key is None:
                endpoint = updates["model_data"]["api_url"]

                if endpoint is None:
                    if wait_for_deployment:
                        continue
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
                   
                    raise ValueError(f"Failed to get model_data_access_key, please check status at: {URL}/oneclick")

            # keep hitting /metadata and see if model is ready or not
            r = requests.get(
                url=f"{endpoint}/metadata", headers={"NBX-KEY": model_data_access_key, "Authorization": f"Bearer {access_token}"}
            )
            if r.status_code == 200:
                console._log(f"Model is ready")
                break

        # if failed exit
        elif "failed" in statuses[-1]["status"]:
            break

    console.stop("Process Complete")
    console.rule()


