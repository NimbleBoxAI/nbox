QuickStart with NBX-Jobs
========================

How do create a new job? Go to your terminal

.. code-block:: bash

  nbx jobs new moonshot


So if I am putting my job in production I will call the command above. ``nbox`` will ask you a couple of questions to reduce your efforts and making it more interactive.
Finally, it will create a folder ``PROJECT_NAME`` with following files in it:

.. code-block::
  
  ./
  ├── .nboxignore
  ├── nbx_user.py
  └── requirements.txt

You can add all your dependencies in the ``requirements.txt`` file which will be our source of truth. Open ``nbx_user.py`` and populate the function ``get_op``:

.. code-block:: python

  def get_op(serving: bool = False) -> Operator:
    if serving:
      # initialise your Operator for NBX-Serving here
      operator = None

      # confused? uncomment lines below
      # from nbox.lib.demo import MagicServing
      # operator = MagicServing()
    else:
      # initialise your Operator for NBX-Job here
      operator = None

      # confused? uncomment lines below
      from nbox.lib.demo import Magic
      operator = Magic()

    return operator


If you are unsure, you can just uncomment the two lines and return the initialised ``Magic`` operator. Go to the folder and run the following commands to deploy:

.. code-block:: bash

  nbx jobs upload moonshot --id_or_name "moonshot" --workspace_id "your-workspace-id"


The logs will contain all the instructions for ``trigger``-ing the job. With the above code you have only uploaded the code, to run the job
you will need to trigger it. This can be done from the dashboard or from the code as given in the logs.


Creating Custom Jobs
--------------------

In NimbleBox, you create jobs using a singular class called ``Operators``, there is nothing else. You can chain them, recurse through them, call them inside each
other or any other code structure that you prefer. You can find documentation `here <nbox_operator.html>`_, fundamentally, you need to do two things, define
the ``__init__`` function and ``forward`` function, just like how you do in case of pytorch.

.. code-block:: python

  from nbox import Operator

  class SomeTask(Operator):
    def __init__(self):
      # >>> YOUR CODE HERE <<< #
      # you would want to perform actions like creating of folders, etc.
      pass

    def forward(self);
      # >>> YOUR CODE HERE <<< #
      # you would want to perform actions like downloading of data, etc.
      pass

  job = SomeTask()

More
----

That's it when you want to run a batch process. Till then you can read `FAQs <jobsdeploy.faq.html>`_.