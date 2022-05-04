QuickStart with NBX-Jobs
========================

How do create a new job? Go to your terminal

.. code-block:: bash

  python3 -m nbox jobs new --help

  # signature
  
  nbox jobs new PROJECT_NAME --workspace_id=WORKSPACE_ID

So if I am putting my job in production I will call the command above. ``nbox`` will ask you a couple of questions to reduce your efforts and making it more interactive.
Finally, it will create a folder ``PROJECT_NAME`` with following files in it:

.. code-block::
  
  ./
  ├── README.md
  ├── exe.py
  ├── nbx_user.py
  └── requirements.txt

You can add all your dependencies in the ``requirements.txt`` file which will be our source of truth. Open ``nbx_user.py`` and populate the function ``get_op``:

.. code-block:: python

  def get_op() -> Operator:
  """Function to initialising your job, it might require passing a bunch of arguments.
  Use this function to get the operator by manually defining things here"""
  # from nbox.nbxlib.ops import Magic
  # return Magic()

  # >>> YOUR CODE HERE <<< #
  return

If you are unsure, you can just uncomment the first two lines and return the initialised ``Magic`` operator. Go to the folder and run the following commands to deploy:

.. code-block:: bash

  cd PROJECT_NAME/      # go the folder
  python3 exe.py deploy # put your job on production.


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


FAQs
----

For FAQs regarding the platform click `here <https://docs.v2.nimblebox.ai/developer-tools/jobs/get-started>`_.

**How to create a new job?**

``python3 -m nbox jobs new --help``

**How do I transfer file to an object store like S3?**

Currently, you can write your own operator or reuse the existing ``boto3`` code that you have. You don't have to learn the intricacies
of our system, you write code the way you like it.

