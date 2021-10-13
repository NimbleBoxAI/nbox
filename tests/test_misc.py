import os
from pprint import pprint
import unittest

import nbox
from nbox import utils

from functools import lru_cache

# Train a model.
import sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


@lru_cache
def get_model_nbox(*args, **kwargs):
    return nbox.load(*args, **kwargs)


@lru_cache
def get_model_sk():
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    clr = RandomForestClassifier()
    clr.fit(X_train, y_train)
    return clr, X_test, y_test


class NboxModelTest(unittest.TestCase):
    def test_model_meta_nbox(self):
        # does this work on nbox models
        image = os.path.join(utils.folder(__file__), "assets/cat.jpg")
        model = get_model_nbox("torchvision/resnet18")
        meta, _ = model.get_nbox_meta(image)
        self.assertEqual(
            meta,
            {
                "inputs": {
                    "input_0": {
                        "dtype": "torch.float32",
                        "name": "input_0",
                        "tensorShape": {
                            "dim": [
                                {"name": "", "size": 1},
                                {"name": "", "size": 3},
                                {"name": "", "size": 720},
                                {"name": "", "size": 1280},
                            ],
                            "unknownRank": False,
                        },
                    }
                },
                "outputs": {
                    "output_0": {
                        "dtype": "torch.float32",
                        "name": "output_0",
                        "tensorShape": {"dim": [{"name": "", "size": 1}, {"name": "", "size": 1000}], "unknownRank": False},
                    }
                },
            },
        )

    def test_model_meta_sk(self):
        # does this work with scikit learn
        model, X_test, y_test = get_model_sk()
        model = nbox.Model(model)
        meta, _ = model.get_nbox_meta(X_test)
        self.assertEqual(
            meta,
            {
                "inputs": {
                    "input_0": {
                        "dtype": "float64",
                        "name": "input_0",
                        "tensorShape": {"dim": [{"name": "", "size": 38}, {"name": "", "size": 4}], "unknownRank": False},
                    }
                },
                "outputs": {
                    "output_0": {
                        "dtype": "int64",
                        "name": "output_0",
                        "tensorShape": {"dim": [{"name": "", "size": 38}], "unknownRank": False},
                    }
                },
            },
        )
