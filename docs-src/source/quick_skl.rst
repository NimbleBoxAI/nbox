
Scikit Learn Models
===================

File: `sk_write.py <https://github.com/NimbleBoxAI/nbox/blob/staging/recipes/sk_write.py>`_

The idea behind ``nbox`` is to simplify and increase the rate of experimentation for ML/DS without having a latency from DevOps teams. For this you can load ``pytorch`` models and ``scikit-learn`` models.

First step is to define any model:

.. code-block:: python

   import sys
   from pprint import pprint as pp

   from sklearn.datasets import load_iris
   from sklearn.model_selection import train_test_split
   from sklearn.ensemble import RandomForestClassifier

   # create a model
   iris = load_iris()
   X, y = iris.data, iris.target
   X_train, X_test, y_train, y_test = train_test_split(X, y)
   clr = RandomForestClassifier()
   clr.fit(X_train, y_train)
   acc = clr.score(X_test, y_test)

Then all you need is to load to ``nbox.Model``\ :

.. code-block:: python

   model = nbox.Model(clr) # <--------- load the model
   y_nbox = model(X_test)  # <--------- predict by passing numpy arrays

   meta, args = model.get_nbox_meta(X) # same method to get the nbox_meta

Finally you can export and deploy this with state of the art simplicity:

.. code-block:: python


   path, _, _ = model.export(X, export_type="onnx")
   model.deploy(X, wait_for_deployment=True, runtime="onnx", deployment_type="nbox")  # <--------- model is deployed
