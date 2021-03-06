import unittest
import tarfile

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import nbox
from nbox import utils

MODEL_TO_LOC = {}

class A_SerialisationTest(unittest.TestCase):
    def test_resnet18_onnx(self):
        model = nbox.load("torchvision/resnet18", pretrained=True)
        model.eval()
        image = "https://media.newyorker.com/photos/590971712179605b11ad7a88/16:9/w_1999,h_1124,c_limit/Jabr-AreCatsDomesticated.jpg"
        nbx_path = model.serialise(input_object = image, model_name = None, export_type = "onnx")
        self.assertTrue(tarfile.is_tarfile(nbx_path))
        MODEL_TO_LOC["A"] = nbx_path

    def test_resnet18_torchscript(self):
        model = nbox.load("torchvision/resnet18", pretrained=True)
        model.eval()
        image = "https://media.newyorker.com/photos/590971712179605b11ad7a88/16:9/w_1999,h_1124,c_limit/Jabr-AreCatsDomesticated.jpg"
        nbx_path = model.serialise(input_object = image, model_name = None, export_type = "torchscript")
        self.assertTrue(tarfile.is_tarfile(nbx_path))
        MODEL_TO_LOC["B"] = nbx_path

    def test_skl_onnx(self):
        # Load the dataset and train a simple model
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        clr = RandomForestClassifier()
        clr.fit(X_train, y_train)

        # boot nbox Model
        model = nbox.Model(clr)
        nbx_path = model.serialise(input_object = X_test, model_name = None, export_type = "onnx")
        self.assertTrue(tarfile.is_tarfile(nbx_path))
        MODEL_TO_LOC["C"] = nbx_path

    def test_skl_pkl(self):
        # Load the dataset and train a simple model
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        clr = RandomForestClassifier()
        clr.fit(X_train, y_train)

        # boot nbox Model
        model = nbox.Model(clr)
        nbx_path = model.serialise(input_object = X_test, model_name = None, export_type = "pkl")
        self.assertTrue(tarfile.is_tarfile(nbx_path))
        MODEL_TO_LOC["D"] = nbx_path

class B_DeserialisationTest(unittest.TestCase):
    @unittest.skipIf(not utils._isthere("onnxruntime"), "ONNX Runtime is not installed")
    def test_resnet18_onnx(self):
        fp = MODEL_TO_LOC["A"]
        model = nbox.Model.deserialise(fp)
        image = "https://media.newyorker.com/photos/590971712179605b11ad7a88/16:9/w_1999,h_1124,c_limit/Jabr-AreCatsDomesticated.jpg"
        out = model(image, return_as_dict = True)
        self.assertEqual(out["output_0"].shape, (1, 1000))

    def test_resnet18_torchscript(self):
        fp = MODEL_TO_LOC["B"]
        model = nbox.Model.deserialise(fp)
        image = "https://media.newyorker.com/photos/590971712179605b11ad7a88/16:9/w_1999,h_1124,c_limit/Jabr-AreCatsDomesticated.jpg"
        out = model(image, return_as_dict = True)
        self.assertEqual(out["output_0"].shape, (1, 1000))

    @unittest.skipIf(not utils._isthere("onnxruntime"), "ONNX Runtime is not installed")
    def test_skl_onnx(self):
        fp = MODEL_TO_LOC["C"]
        model = nbox.Model.deserialise(fp)
        X = load_iris().data
        out = model(X, return_as_dict = True)
        self.assertEqual(out["output_0"].shape, (150,))

    def test_skl_pkl(self):
        fp = MODEL_TO_LOC["D"]
        model = nbox.Model.deserialise(fp)
        X = load_iris().data
        out = model(X, return_as_dict = True)
        self.assertEqual(out["output_0"].shape, (150,))

class C_DeserialisationTest(unittest.TestCase):
    @unittest.skipIf(not utils._isthere("onnxruntime"), "ONNX Runtime is not installed")
    def test_resnet18_onnx(self):
        fp = MODEL_TO_LOC["A"]
        model = nbox.Model.deserialise(fp)
        image = "https://media.newyorker.com/photos/590971712179605b11ad7a88/16:9/w_1999,h_1124,c_limit/Jabr-AreCatsDomesticated.jpg"
        out = model(image, return_as_dict = False)
        scores = out[0]
        self.assertEqual(scores.shape, (1, 1000))

    def test_resnet18_torchscript(self):
        fp = MODEL_TO_LOC["B"]
        model = nbox.Model.deserialise(fp)
        image = "https://media.newyorker.com/photos/590971712179605b11ad7a88/16:9/w_1999,h_1124,c_limit/Jabr-AreCatsDomesticated.jpg"
        out = model(image, return_as_dict = False)
        self.assertEqual(out.shape, (1, 1000))

    @unittest.skipIf(not utils._isthere("onnxruntime"), "ONNX Runtime is not installed")
    def test_skl_onnx(self):
        fp = MODEL_TO_LOC["C"]
        model = nbox.Model.deserialise(fp)
        X = load_iris().data
        out = model(X, return_as_dict = False)
        output_labels, output_probabs = out
        self.assertEqual(output_labels.shape, (150,))

    def test_skl_pkl(self):
        fp = MODEL_TO_LOC["D"]
        model = nbox.Model.deserialise(fp)
        X = load_iris().data
        out = model(X, return_as_dict = False)
        self.assertEqual(out.shape, (150,))
