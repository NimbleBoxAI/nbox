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
  
  def test_cloud_infer_resnet(self):
    cache_dir = os.path.join(utils.folder(__file__), "__ignore/")
    os.makedirs(cache_dir, exist_ok = True)
    image = os.path.join(utils.folder(__file__), "assets/cat.jpg")
    model = nbox.NBXApi(
      model_key_or_url="https://api.test-2.nimblebox.ai/yash_bonde_139/cobalt_cricket_4bf1",
      nbx_api_key="",
      category = "image"
    )
    print(model)

    out, headers, t = model(image)
    print("took:", t, "seconds")
    out = np.array(out["outputs"])
    print(out.shape)
    print(headers)
