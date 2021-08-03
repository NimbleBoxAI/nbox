import os
import unittest

import aibox
from aibox import utils

class ImportTest(unittest.TestCase):

    def test_hf_import(self):
        cache_dir = os.path.join(utils.folder(__file__), "__ignore/")
        os.makedirs(cache_dir, exist_ok = True)
        aibox.load(
          "transformers/sshleifer/tiny-gpt2::AutoModelForCausalLM::generation",
          cache_dir = cache_dir
        )
    
    def test_hf_generation(self):
        cache_dir = os.path.join(utils.folder(__file__), "__ignore/")
        os.makedirs(cache_dir, exist_ok = True)
        model = aibox.load(
          "transformers/sshleifer/tiny-gpt2::AutoModelForCausalLM::generation",
          cache_dir = cache_dir
        )
        model.generate(...)

    def test_hf_masked_lm(self):
        cache_dir = os.path.join(utils.folder(__file__), "__ignore/")
        os.makedirs(cache_dir, exist_ok = True)
        model = aibox.load(
          "transformers/sshleifer/tiny-gpt2::AutoModelForCausalLM::generation",
          cache_dir = cache_dir
        )
        model.get(...)

if __name__ == '__main__':
    unittest.main()
