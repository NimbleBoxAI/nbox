import os
import unittest
from functools import lru_cache

import nbox
from nbox import utils


@lru_cache
def get_model(*args, **kwargs):
    return nbox.load(*args, **kwargs)


class DeployTest(unittest.TestCase):

    # def test_deploy_hf_bert(self):
    #     model_key = "transformers/prajjwal1/bert-tiny::AutoModelForMaskedLM"
    #     cache_dir = os.path.join(utils.folder(__file__), "__ignore/")
    #     os.makedirs(cache_dir, exist_ok=True)
    #     model = get_model(model_key, cache_dir=cache_dir)
    #     # out = model("hello world")
    #     # print(type(out), isinstance(out, dict))
    #     url, model_data_access_key = model.deploy(
    #         "hello world",
    #         username=os.getenv("NBX_USERNAME"),
    #         password=os.getenv("NBX_PASSWORD"),
    #         cache_dir=cache_dir,
    #     )

    # def test_deploy_hf_bert_mid(self):
    #     model_key = "transformers/bert-base-uncased::AutoModelForMaskedLM"
    #     cache_dir = os.path.join(utils.folder(__file__), "__ignore/")
    #     os.makedirs(cache_dir, exist_ok=True)
    #     model = get_model(model_key, cache_dir=cache_dir)
    #     # out = model("hello world")
    #     # print(type(out), isinstance(out, dict))
    #     url, model_data_access_key = model.deploy(
    #         "hello world",
    #         username=os.getenv("NBX_USERNAME"),
    #         password=os.getenv("NBX_PASSWORD"),
    #         cache_dir=cache_dir,
    #     )

    # def test_deploy_tv_resnet18(self):
    #   model_key = "torchvision/resnet18"
    #   cache_dir = os.path.join(utils.folder(__file__), "__ignore/")
    #   os.makedirs(cache_dir, exist_ok = True)
    #   image = os.path.join(utils.folder(__file__), "assets/cat.jpg")
    #   model = get_model(model_key)
    #   url, model_data_access_key = model.deploy(
    #     image,
    #     username=os.getenv("NBX_USERNAME"),
    #     password=os.getenv("NBX_PASSWORD"),
    #     cache_dir = cache_dir,
    #   )

    pass
