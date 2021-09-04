import os
import unittest

import nbox
from nbox import utils

from functools import lru_cache


@lru_cache
def get_model(*args, **kwargs):
    return nbox.load(*args, **kwargs)


class ImportTest(unittest.TestCase):
    def test_hf_import(self):
        cache_dir = os.path.join(utils.folder(__file__), "__ignore/")
        os.makedirs(cache_dir, exist_ok=True)
        get_model("transformers/sshleifer/tiny-gpt2::AutoModelForCausalLM::generation", cache_dir=cache_dir)

    def test_hf_string(self):
        cache_dir = os.path.join(utils.folder(__file__), "__ignore/")
        os.makedirs(cache_dir, exist_ok=True)
        model = get_model("transformers/sshleifer/tiny-gpt2::AutoModelForCausalLM::generation", cache_dir=cache_dir)
        model.eval()
        out = model("Hello world")

        self.assertEqual(out.logits.topk(4).indices.tolist(), [[[16046, 17192, 38361, 43423], [16046, 17192, 38361, 43423]]])

    def test_hf_numpy(self):
        import numpy as np

        cache_dir = os.path.join(utils.folder(__file__), "__ignore/")
        os.makedirs(cache_dir, exist_ok=True)
        model = get_model("transformers/sshleifer/tiny-gpt2::AutoModelForCausalLM::generation", cache_dir=cache_dir)
        out = model(np.random.randint(low=0, high=100, size=(12,)))

    def test_hf_string_batch(self):
        cache_dir = os.path.join(utils.folder(__file__), "__ignore/")
        os.makedirs(cache_dir, exist_ok=True)
        model = get_model("transformers/sshleifer/tiny-gpt2::AutoModelForCausalLM::generation", cache_dir=cache_dir)
        out = model(["Hello world", "my foot"])

    def test_hf_masked_lm(self):
        cache_dir = os.path.join(utils.folder(__file__), "__ignore/")
        os.makedirs(cache_dir, exist_ok=True)
        model = nbox.load("transformers/prajjwal1/bert-tiny::AutoModelForMaskedLM", cache_dir=cache_dir)
        out = model("hello world")
        self.assertEqual(out.logits.argmax(-1).tolist(), [[1012, 7592, 2088, 1012]])


if __name__ == "__main__":
    unittest.main()
