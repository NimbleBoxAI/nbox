How to nboxCLI
==============

``nbox`` provides a simple and effective CLI commands that you can add to your current CI/CD pipeline.

As of this writing there is only one task that ``nbox`` CLI does, deployment of models. It directly calls the
``nbox.network.deploy_model`` method along, if needed with authentication for single line in your CI/CD.
When in an automated pipeline, file ``secrets.json`` might not be available so set ``NBX_AUTH=1`` when using CLI.

.. code-block:: bash
   
   NBX_AUTH=1 python -m nbox deploy --model_path=my/model.onnx --deployment_type="nbox"

We use ``fire`` for converting the below method to CLI so a quick go through of the method will tell you
enough about the input. All you need is to convert each ``kwarg`` to ``--kwarg`` in your CLI.

Methods
-------

.. automodule:: nbox.cli
   :members:
   :undoc-members:
   :show-inheritance:
