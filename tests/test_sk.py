import os
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
def get_model():
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    clr = RandomForestClassifier()
    clr.fit(X_train, y_train)
    return clr, X_test, y_test


class ImportTest(unittest.TestCase):
    def test_import(self):
        _m, _x, _y = get_model()
        model = nbox.Model(_m)
        out = model(_x)
        self.assertEqual(out.shape, _y.shape)

    @unittest.expectedFailure
    def test_incorrect_input(Self):
        _m, _x, _y = get_model()
        model = nbox.Model(_m)
        out = model({"input": _x})
