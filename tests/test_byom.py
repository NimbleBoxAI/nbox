import os
import torch
import unittest
import numpy as np
from functools import lru_cache

import nbox
from nbox import utils


@lru_cache()
def get_model_template(spec: str):
    if spec == "I-I":
        # this is an image image model, we hard code to 3 inputs
        class NHeadModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # input_size = (10, 10, 3)
                self.a = torch.nn.Linear(300, 3)

                # input_size = (5, 5, 3)
                self.b = torch.nn.Linear(75, 3)

            def forward(self, a, b):
                return self.a(a.reshape(a.shape[0], -1)) + self.b(b.reshape(b.shape[0], -1))

        # load the model
        model = nbox.Model(NHeadModel(), category={"a": "image", "b": "image"})
        return model, {"a": [1, 3, 10, 10], "b": [1, 3, 5, 5]}

    elif spec == "I-T":
        raise NotImplementedError("TODO")


class PytorchModelLoader(unittest.TestCase):
    def test_image_image_model(self):
        model, templates = get_model_template("I-I")

        # define parser
        parser = nbox.ImageParser(
            post_proc_fn=None,
            templates=templates,
        )
        image = os.path.join(utils.folder(__file__), "assets/cat.jpg")

        # simple fp list -> should fail
        with self.assertRaises(Exception):
            out = parser([image, image])

        # simple dict with 1 input
        out = parser({"a": image, "b": image})
        self.assertEqual(
            {k: v.shape for k, v in out.items()},
            {
                "a": (1, 3, 10, 10),
                "b": (1, 3, 5, 5),
            },
        )

        # simple dict with 3 input
        out = parser({"a": [image, image, image], "b": [image, image, image]})
        self.assertEqual(
            {k: v.shape for k, v in out.items()},
            {
                "a": (3, 3, 10, 10),
                "b": (3, 3, 5, 5),
            },
        )

        # pass to the model
        out = model(out)
        self.assertEqual(
            out.shape,
            (3, 3),
        )
