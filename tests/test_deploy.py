import os
import unittest
from functools import lru_cache

import nbox
from nbox import utils


@lru_cache
def get_model(*args, **kwargs):
    return nbox.load(*args, **kwargs)


class DeployTest(unittest.TestCase):

    # def test_onnx_export_tv(self):
    #   # test onnx export for torchvision models
    #   import torch
    #   model = nbox.load("torchvision/resnet18")
    #   image = os.path.join(utils.folder(__file__), "assets/cat.jpg")
    #   model(image) # pass a file once
    #   print("Input Shape:", model._sample_input.shape)
    #   pt_model = model.model.eval()

    #   torch.onnx.export(
    #     pt_model,
    #     model._sample_input,
    #     os.path.join(utils.folder(__file__), "assets/sample.onnx"),
    #     verbose=True,
    #   )

    # def test_onnx_export_hf_bert(self):
    #   model_key = "transformers/prajjwal1/bert-tiny::AutoModelForMaskedLM"
    #   cache_dir = os.path.join(utils.folder(__file__), "__ignore/")
    #   os.makedirs(cache_dir, exist_ok = True)
    #   model = get_model(model_key,cache_dir = cache_dir)
    #   out = model("hello world")
    #   print(type(out), isinstance(out, dict))

    def test_deploy_hf_bert(self):
        model_key = "transformers/prajjwal1/bert-tiny::AutoModelForMaskedLM"
        cache_dir = os.path.join(utils.folder(__file__), "__ignore/")
        os.makedirs(cache_dir, exist_ok=True)
        model = get_model(model_key, cache_dir=cache_dir)
        # out = model("hello world")
        # print(type(out), isinstance(out, dict))
        url = model.deploy(
            "hello world",
            username="",
            password="",
            cache_dir=cache_dir,
        )

    # def test_deploy_tv_resnet18(self):
    #   model_key = "torchvision/resnet18"
    #   cache_dir = os.path.join(utils.folder(__file__), "__ignore/")
    #   os.makedirs(cache_dir, exist_ok = True)
    #   image = os.path.join(utils.folder(__file__), "assets/cat.jpg")
    #   model = get_model(model_key)
    #   url = model.deploy(
    #     image,
    #     username = "",
    #     password = "",
    #     cache_dir = cache_dir,
    #   )
