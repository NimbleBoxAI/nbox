QuickStart with NBX-Deploy
==========================

Deployment is a major bottleneck for DS/ML engineers to get their systems into production. Unless any
model is in production there really is no way to complete a project. ``nbox`` as an SDK makes deployment just one
command, **literally**\ ! 

Till now you have see how to deploy a batch process, if not you might want to first complete `jobs tutorial <jobsdeploy.job>`_
first. Let's continue with our ``moonshot`` project.

Uncomment Two lines
-------------------

Open ``nbx_user.py`` and uncomment the lines when ``serving == True`` so the function will return the ``MagicServing`` operator.

.. code-block:: python

  def get_op(serving: bool = False) -> Operator:
    if serving:
      # initialise your Operator for NBX-Serving here
      operator = None

      # confused? uncomment lines below
      from nbox.lib.demo import MagicServing
      operator = MagicServing()
    else:
      # initialise your Operator for NBX-Job here
      operator = None

      # confused? uncomment lines below
      # from nbox.lib.demo import Magic
      # operator = Magic()

    return operator

To upload we use the exact same code.

.. code-block:: bash

  nbx serve upload moonshot --id_or_name "moonshot_api" --workspace_id "your-workspace-id"

The logs will contain instructions for using the API endpoint.

Creating Custom Jobs
--------------------

In NimbleBox, you create deployments using a singular class called ``Operators``. You can find documentation `here <nbox_operator.html>`_,
fundamentally, you need to do two things, define the ``__init__`` function and ``forward`` function, just like how you do in case of pytorch.
**However there's a catch.** Beaware that object returned from the ``forward`` function is JSON serialisable. In order to make it perform
even better add annotations to the input arguments.

.. code-block:: python

   def forward(self, name: str):
     
     # invalid return since cannot be JSON serialised
     return "Hello " + name + "!"

     # valid return since can be JSON serialised
     return {
      "data": "Hello " + name + "!"
     }


More
----

That's it when you want to run an API endpoint. Till then you can read `FAQs <jobsdeploy.faq.html>`_.

