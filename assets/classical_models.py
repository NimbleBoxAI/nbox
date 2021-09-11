#!/usr/bin/env python

import numpy as np

# Train a model.
from re import A
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y)

print("Training a random forest...")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

clr = RandomForestClassifier()
clr.fit(X_train, y_train)

print("Accuracy on training set:", clr.score(X_train, y_train))
print("Accuracy on test set:", clr.score(X_test, y_test))


# Convert into ONNX format
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

print("Converting into ONNX format...")
initial_type = [("float_input", FloatTensorType([None, 4]))]
onx = convert_sklearn(clr, initial_types=initial_type)
with open("rf_iris.onnx", "wb") as f:
    print("Writing to file...", "rf_iris.onnx")
    f.write(onx.SerializeToString())


# Compute the prediction with ONNX Runtime
import onnxruntime as rt
import numpy
print("Compiling model to ONNX Runtime")
sess = rt.InferenceSession("rf_iris.onnx")
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
pred_onx = sess.run([label_name], {input_name: X_test.astype(numpy.float32)})[0]

print(pred_onx.shape)
print("Accuracy with ONNX export:", sum(pred_onx == y_test) / len(y_test))

np.allclose(pred_onx, y_test)
