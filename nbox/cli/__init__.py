import json
import os

import requests
from nbox.model import Model
from nbox.utils import join


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
    
def deploy(model_file,input_object,model_key, model_name = None,
        deployment_type = "ovms2",category=None, tokenizer=None,  model_meta = None,verbose = False, 
        cache_dir = None,
        wait_for_deployment = False):

    model =  Model(
        model_or_model_url=model_file,
        category=json.loads(category or "{}") or None,
        tokenizer=tokenizer,
        model_key= model_key,
        model_meta = model_meta,
        verbose = verbose,
    )
    return model.deploy(input_object=input_object,model_name=model_name,cache_dir=cache_dir,wait_for_deployment=wait_for_deployment,deployment_type=deployment_type)



