import sys
from pprint import pprint as pp

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

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

path, _, _ = model.export(X, export_type="onnx")
print("output path:", path)
hr()

model.deploy(X, wait_for_deployment=True, deployment_type="nbox")  # <--------- model is deployed
