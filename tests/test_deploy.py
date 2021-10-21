import os
from tests.test_cloud_infer import SKIP_CONDITION
import unittest
from functools import lru_cache
from datetime import datetime

import nbox
from nbox import utils

# Train a model.
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

SKIP_CONDITION = os.getenv("NBOX_TEST_DEPLOY", False) != False


@lru_cache
def get_model(*args, **kwargs):
    return nbox.load(*args, **kwargs)


@lru_cache
def get_model_sk():
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    clr = RandomForestClassifier()
    clr.fit(X_train, y_train)
    return clr, X_test, y_test


class DeployTest(unittest.TestCase):
    @unittest.skipUnless(SKIP_CONDITION, "Skipping deployment test")
    def test_deploy_tv_resnet18_ovms2(self):
        model_key = "torchvision/resnet18"
        cache_dir = os.path.join(utils.folder(__file__), "__ignore/")
        os.makedirs(cache_dir, exist_ok=True)
        image = os.path.join(utils.folder(__file__), "assets/cat.jpg")
        model = get_model(model_key)
        url, model_data_access_key = model.deploy(
            image,
            model_name="dut_r18_ovms2_{}".format(datetime.now().strftime("%m%d")),
            cache_dir=cache_dir,
            deployment_type="ovms2",
            # runtime="onnx", # auto converts to onnx
            wait_for_deployment=False,
        )

    @unittest.skipUnless(SKIP_CONDITION, "Skipping deployment test")
    def test_deploy_tv_resnet18_nbox_onnx(self):
        model_key = "torchvision/resnet18"
        cache_dir = os.path.join(utils.folder(__file__), "__ignore/")
        os.makedirs(cache_dir, exist_ok=True)
        image = os.path.join(utils.folder(__file__), "assets/cat.jpg")
        model = get_model(model_key)
        url, model_data_access_key = model.deploy(
            image,
            model_name="dut_r18_nbox_onnx_{}".format(datetime.now().strftime("%m%d")),
            cache_dir=cache_dir,
            deployment_type="nbox",
            runtime="onnx",
            wait_for_deployment=False,
        )

    @unittest.skipUnless(SKIP_CONDITION, "Skipping deployment test")
    def test_deploy_tv_resnet18_nbox_torchscript(self):
        model_key = "torchvision/resnet18"
        cache_dir = os.path.join(utils.folder(__file__), "__ignore/")
        os.makedirs(cache_dir, exist_ok=True)
        image = os.path.join(utils.folder(__file__), "assets/cat.jpg")
        model = get_model(model_key)
        url, model_data_access_key = model.deploy(
            image,
            model_name="dut_r18_nbox_trchscrpt_{}".format(datetime.now().strftime("%m%d")),
            cache_dir=cache_dir,
            deployment_type="nbox",
            runtime="torchscript",
            wait_for_deployment=False,
        )

    @unittest.skipUnless(SKIP_CONDITION, "Skipping deployment test")
    def test_deploy_tv_randomforest_nbox_onnx(self):
        _m, _x, _y = get_model_sk()
        cache_dir = os.path.join(utils.folder(__file__), "__ignore/")
        model = nbox.Model(_m)
        url, model_data_access_key = model.deploy(
            _x,
            model_name="dut_sk_nbox_onnx_{}".format(datetime.now().strftime("%m%d")),
            cache_dir=cache_dir,
            deployment_type="nbox",
            runtime="onnx",
            wait_for_deployment=False,
        )

    @unittest.skipUnless(SKIP_CONDITION, "Skipping deployment test")
    def test_deploy_tv_randomforest_nbox_pkl(self):
        _m, _x, _y = get_model_sk()
        cache_dir = os.path.join(utils.folder(__file__), "__ignore/")
        model = nbox.Model(_m)
        url, model_data_access_key = model.deploy(
            _x,
            model_name="dut_sk_nbox_pkl_{}".format(datetime.now().strftime("%m%d")),
            cache_dir=cache_dir,
            deployment_type="nbox",
            runtime="pkl",
            wait_for_deployment=False,
        )

    # @unittest.skipUnless(SKIP_CONDITION, "Skipping deployment test")
    # def test_deploy_cli(self):
    #     cli_command = """python3 -m nbox deploy --model_path"""
