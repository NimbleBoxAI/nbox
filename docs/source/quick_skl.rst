
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

   model = nbox.Model(clr) # load the model
   y_nbox = model(X_test)  # predict by passing numpy arrays

   meta, args = model.get_nbox_meta(X) # same method to get the nbox_meta

By default we call ``.predict()`` method, however you might want to call ``.predict_proba()``
in that case you can override by telling with ``method`` keyword as follows:

.. code-block:: python

   out = model(X, method="predict") # default
   out = model(X, method="predict_log_proba")
   out = model(X, method="predict_proba")

Finally you can export or deploy this with state of the art simplicity:

.. code-block:: python

   # export the model as "onnx"
   path, _, _ = model.export(X, export_type="onnx")
   
   # deploy the model on our managed kubernetes
   # written as a first class method to "nbox.Model" for paramount simplicity
   model.deploy(X, runtime="onnx", deployment_type="nbox")

Once the model is deployed you can directly load and use the model as follows:

.. code-block:: python

   # load the model
   cloud_model = nbox.load(url, key)

   # use cloud model along with "method" keyword for specific 
   out = cloud_model(X, method="predict")
   out = cloud_model(X, method="predict_log_proba")
   out = cloud_model(X, method="predict_proba")

