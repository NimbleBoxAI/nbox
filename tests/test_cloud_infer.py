import os
import unittest
import numpy as np
from functools import lru_cache

import nbox
from nbox import utils


_a = os.environ.get("NBOX_TEST_CLOUD_INFER", False) != False
_b = os.environ.get("NBOX_API_KEY", False) != False
SKIP_CONDITION = _a & _b


@lru_cache
def get_model(*args, **kwargs):
    return nbox.load(*args, **kwargs)


class CloudInferTest(unittest.TestCase):
    @unittest.skipUnless(SKIP_CONDITION, f"Skip: {(_a, _b)}")
    def test_cloud_infer_resnet(self):
        model = nbox.load(
            model_key_or_url="https://api.nimblebox.ai/yash_bonde_139/brownian_perimeter_4bf1/",
            nbx_api_key=os.environ.get("NBOX_API_KEY"),
            verbose=True,
        )

        cache_dir = os.path.join(utils.folder(__file__), "__ignore/")
        os.makedirs(cache_dir, exist_ok=True)

        # model consuming image path
        image = os.path.join(utils.folder(__file__), "assets/cat.jpg")
        out = model(image)
        self.assertEquals(tuple(out.shape), (1, 1000))

        # model consuming URL
        out = model(
            "https://www.cnet.com/a/img/CSTqzAl5wJ57HHyASLD-a0vS2O0=/940x528/2021/04/05/9e065d90-51f2-46c5-bd3a-416fd4983c1a/elantra-1080p.jpg",
        )
        self.assertNotIn("error", out)
        self.assertEquals(tuple(out.shape), (1, 1000))

        # model consuming URL list
        out = model(
            [
                "https://www.cnet.com/a/img/CSTqzAl5wJ57HHyASLD-a0vS2O0=/940x528/2021/04/05/9e065d90-51f2-46c5-bd3a-416fd4983c1a/elantra-1080p.jpg"
                for _ in range(2)
            ]
        )
        self.assertNotIn("error", out)
        self.assertEquals(tuple(out.shape), (2, 1000))

    pass
