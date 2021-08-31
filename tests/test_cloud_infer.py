import os
import unittest
import numpy as np
from functools import lru_cache

from requests.api import head

import nbox
from nbox import utils


@lru_cache
def get_model(*args, **kwargs):
    return nbox.load(*args, **kwargs)


class CloudInferTest(unittest.TestCase):

    # def test_cloud_infer_resnet(self):
    #   cache_dir = os.path.join(utils.folder(__file__), "__ignore/")
    #   os.makedirs(cache_dir, exist_ok = True)
    #   image = os.path.join(utils.folder(__file__), "assets/cat.jpg")
    #   model = nbox.NBXApi(
    #     model_key_or_url="",
    #     nbx_api_key="",
    #     category = "image"
    #   )
    #   print(model)
    #   out, headers, t = model(image)
    #   print("took:", t, "seconds")
    #   out = np.array(out["outputs"])
    #   print(out.argmax(-1))
    #   print(headers)
    #   out, headers, t = model("https://www.cnet.com/a/img/CSTqzAl5wJ57HHyASLD-a0vS2O0=/940x528/2021/04/05/9e065d90-51f2-46c5-bd3a-416fd4983c1a/elantra-1080p.jpg")
    #   print("took:", t, "seconds")
    #   out = np.array(out["outputs"])
    #   print(out.argmax(-1))
    #   print(headers)

    # def test_cloud_infer_bert_mid(self):
    #   model = nbox.NBXApi(
    #     model_key_or_url="",
    #     nbx_api_key="",
    #     category = "text"
    #   )
    #   print(model)

    pass
