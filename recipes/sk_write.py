import sys
<<<<<<< HEAD
import numpy as np
=======
>>>>>>> master
from pprint import pprint as pp

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
<<<<<<< HEAD
from transformers.utils.dummy_pt_objects import XLMForTokenClassification
=======
>>>>>>> master

# ---------
import nbox

# ---------


def hr():
    print("-" * 80)


# create a model
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y)
clr = RandomForestClassifier()
clr.fit(X_train, y_train)
acc = clr.score(X_test, y_test)
hr()

model = nbox.Model(clr)  # <--------- load the model
y_nbox = model(X_test)  # <--------- predict by passing numpy arrays

print("Accuracy (vanilla):", acc)
print("             Model:", model)
print("    Output (shape):", y_nbox.shape)
print("     Output (type):", y_nbox.dtype)
print("   Accuracy (nbox):", (y_nbox == y_test).mean())
hr()

meta, args = model.get_nbox_meta(X)
pp(meta)

out = model(X, method="predict_proba")
print(out.shape)

# print(model.model_or_model_url.predict_proba(X).shape)
# print(np.array(model.model_or_model_url.predict_log_proba(X).tolist()))

path, _, _ = model.export(X, export_type="pkl")
print("output path:", path)
hr()

# We are using `nbox` as the server type and `onnx` as the runtime
# read more: https://nimbleboxai.github.io/nbox/nbox.model.html
url, key = model.deploy(X, wait_for_deployment=True, runtime = "pkl", deployment_type="nbox")

# In RandomForest there are 3 different methods of forward pass that can be used
# predict(X):          Predict class for X.
# predict_log_proba(X) Predict class log-probabilities for X.
# predict_proba(X)     Predict class probabilities for X.
# By default I use `predict` and in order to use these over the api you will have to
# tell which method you want to call
cloud_model = nbox.load(url, key)
out = cloud_model(X, method="predict")
out = cloud_model(X, method="predict_log_proba")
out = cloud_model(X, method="predict_proba")
