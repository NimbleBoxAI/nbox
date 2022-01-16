import os
import unittest

import nbox
from nbox import utils

from functools import lru_cache


@lru_cache
def get_model(*args, **kwargs):
    return nbox.load(*args, **kwargs)


@lru_cache
def get_parser(*args, **kwargs):
    model = get_model(*args, **kwargs)
    return model.text_parser


class ImportTest(unittest.TestCase):
    def test_hf_string(self):
        cache_dir = os.path.join(utils.folder(__file__), "__ignore/")
        os.makedirs(cache_dir, exist_ok=True)
        model = get_model("transformers/sshleifer/tiny-gpt2::AutoModelForCausalLM", cache_dir=cache_dir)
        model.eval()
        out = model("Hello world")
        self.assertEqual(out.logits.topk(4).indices.tolist(), [[[16046, 17192, 38361, 43423], [16046, 17192, 38361, 43423]]])

    def test_hf_string_batch(self):
        cache_dir = os.path.join(utils.folder(__file__), "__ignore/")
        os.makedirs(cache_dir, exist_ok=True)
        model = get_model("transformers/sshleifer/tiny-gpt2::AutoModelForCausalLM", cache_dir=cache_dir)
        out = model(["Hello world", "my foot"])
        self.assertEqual(out.logits.argmax(-1).tolist(), [[16046, 16046], [16046, 16046]])

    def test_hf_masked_lm(self):
        cache_dir = os.path.join(utils.folder(__file__), "__ignore/")
        os.makedirs(cache_dir, exist_ok=True)
        model = nbox.load("transformers/prajjwal1/bert-tiny::AutoModelForMaskedLM", cache_dir=cache_dir)
        out = model("hello world")
        self.assertEqual(out.logits.argmax(-1).tolist(), [[1012, 7592, 2088, 1012]])

    def test_hf_numpy(self):
        import numpy as np

        cache_dir = os.path.join(utils.folder(__file__), "__ignore/")
        os.makedirs(cache_dir, exist_ok=True)
        model = get_model("transformers/sshleifer/tiny-gpt2::AutoModelForCausalLM", cache_dir=cache_dir)
        out = model(np.array([[0, 1, 1, 2, 4, 5, 6, 6, 7, 8, 0]]))
        self.assertEqual(out.logits.argmax(-1).tolist(), [[16046, 16046, 16046, 5087, 16046, 16046, 5087, 5087, 16046, 16046, 16046]])


class ParserTest(unittest.TestCase):
    def test_string(self):
        cache_dir = os.path.join(utils.folder(__file__), "__ignore/")
        os.makedirs(cache_dir, exist_ok=True)
        parser = get_parser("transformers/sshleifer/tiny-gpt2::AutoModelForCausalLM", cache_dir=cache_dir)
        out = parser("hello world")

        self.assertEqual(
            {
                "input_ids": list(out["input_ids"].shape),
                "attention_mask": list(out["attention_mask"].shape),
            },
            {
                "input_ids": [1, 2],
                "attention_mask": [1, 2],
            },
        )

    def test_list_string(self):
        cache_dir = os.path.join(utils.folder(__file__), "__ignore/")
        os.makedirs(cache_dir, exist_ok=True)
        parser = get_parser("transformers/sshleifer/tiny-gpt2::AutoModelForCausalLM", cache_dir=cache_dir)
        out = parser(["wabba lubba dub dub", "BoJack Horseman - A wise man told me that", "I can't believe you just did that!"])

        self.assertEqual(
            {
                "input_ids": list(out["input_ids"].shape),
                "attention_mask": list(out["attention_mask"].shape),
            },
            {
                "input_ids": [3, 11],
                "attention_mask": [3, 11],
            },
        )

    def test_dict_strings(self):
        cache_dir = os.path.join(utils.folder(__file__), "__ignore/")
        os.makedirs(cache_dir, exist_ok=True)
        parser = get_parser("transformers/sshleifer/tiny-gpt2::AutoModelForCausalLM", cache_dir=cache_dir)
        out = parser({"input_sentence": "hello world", "target_sentence": "wabba lubba dub dub"})

        self.assertEqual(
            {
                "input_sentence": {
                    "input_ids": list(out["input_sentence"]["input_ids"].shape),
                    "attention_mask": list(out["input_sentence"]["attention_mask"].shape),
                },
                "target_sentence": {
                    "input_ids": list(out["target_sentence"]["input_ids"].shape),
                    "attention_mask": list(out["target_sentence"]["attention_mask"].shape),
                },
            },
            {
                "input_sentence": {
                    "input_ids": [1, 2],
                    "attention_mask": [1, 2],
                },
                "target_sentence": {
                    "input_ids": [1, 7],
                    "attention_mask": [1, 7],
                },
            },
        )

    def test_dict_list_strings(self):
        cache_dir = os.path.join(utils.folder(__file__), "__ignore/")
        os.makedirs(cache_dir, exist_ok=True)
        parser = get_parser("transformers/sshleifer/tiny-gpt2::AutoModelForCausalLM", cache_dir=cache_dir)
        out = parser(
            {"input_sentence": "hello world", "target_sentences": ["wabba lubba dub dub", "BoJack Horseman - A wise man told me that"]}
        )

        self.assertEqual(
            {
                "input_sentence": {
                    "input_ids": list(out["input_sentence"]["input_ids"].shape),
                    "attention_mask": list(out["input_sentence"]["attention_mask"].shape),
                },
                "target_sentences": {
                    "input_ids": list(out["target_sentences"]["input_ids"].shape),
                    "attention_mask": list(out["target_sentences"]["attention_mask"].shape),
                },
            },
            {
                "input_sentence": {"input_ids": [1, 2], "attention_mask": [1, 2]},
                "target_sentences": {"input_ids": [2, 11], "attention_mask": [2, 11]},
            },
        )

