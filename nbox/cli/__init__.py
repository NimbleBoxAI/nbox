import json
import os

from nbox.utils import join
from nbox.network import one_click_deploy
from nbox.user import get_access_token, create_secret_file, reinit_secret


def login(username, password, nbx_home_url="https://nimblebox.ai"):
    """Login NimbleBox.ai account

    Args:
        username (str): Your NimbleBox.ai username
        password (str): Your NimbleBox.ai password
        nbx_home_url (str, optional): Do not use, this is used for debugging
    """
    access_token = get_access_token(nbx_home_url, username, password)
    create_secret_file(username, access_token, nbx_home_url)
    reinit_secret()  # reintialize secret variable as it will be used everywhere


def deploy(
    config_path: str = None,
    model_path: str = None,
    model_name: str = None,
    nbox_meta: str = None,
    deployment_type: str = None,
    convert_args: str = None,
    wait_for_deployment: bool = False,
    print_in_logs: bool = False,
):
    """Deploy a model from nbox CLI. Add this to your actions and see the magic happen!
    If you are using a config file then data will be loaded from there and other kwargs will be ignored.

    Args:
        config_path (str, optional): path to your config file
        model_path (str, optional): path to your model
        model_name (str, optional): name of your model
        nbox_meta ([str, dict], optional): path to your nbox_meta json file or can be a dict if using config_path
        deployment_type (str, optional): type of deployment, can be one of: ovms2, onnx-rt, nbox
        convert_args (str, optional): if using ovms2 deployment type, you must pass convertion CLI args
        wait_for_deployment (bool, optional): wait for deployment to finish, if False this behaves async
        print_in_logs (bool, optional): print logs in stdout

    Raises:
        Exception: if deployment type is not supported
        Exception: if model path is not found
        Exception: if nbox_meta is incorrect
        Exception: if deployment_type == "ovms2" but convert_args is not provided
    """
    if not os.path.exists(join(os.path.expanduser("~"), ".nbx")):
        return "You are not logged in try nbox login --help"

    if config_path != None:
        with open(config_path, "r") as f:
            config = json.load(f)
        deploy(**config)

    else:
        # is model path valid and given
        if not os.path.exists(model_path):
            raise Exception(f"Model path {model_path} does not exist")

        # check if nbox_meta is correct
        if isinstance(nbox_meta, str):
            if not os.path.exists(nbox_meta):
                raise Exception(f"Nbox meta path {nbox_meta} does not exist. see nbox.Model.get_nbox_meta()")
            with open(nbox_meta, "r") as f:
                nbox_meta = json.load(f)
        else:
            assert isinstance(nbox_meta, dict), "nbox_meta must be a dict"

        # validation of deployment_type
        assert deployment_type in ["ovms2", "onnx-rt", "nbox"], "Deployment type must be one of: ovms2, onnx-rt, nbox"
        if deployment_type == "ovms2":
            assert convert_args is not None, (
                "Please provide convert args when using OVMS deployment, "
                "use nbox.Model.deploy(deployment_type == 'ovms2') if you are unsure!"
            )

        # one click deploy
        endpoint, model_data_access_key = one_click_deploy(
            model_path,
            deployment_type,
            nbox_meta,
            model_name,
            wait_for_deployment,
            convert_args,
        )

        # print to logs if needed
        if wait_for_deployment and print_in_logs:
            print("             endpoint:", endpoint)
            print("model_data_access_key:", model_data_access_key)
