Jobs + Deploy FAQs
==================

For FAQs regarding the platform click `here <https://docs.v2.nimblebox.ai/developer-tools/jobs/get-started>`_.

**How can I change resources?**

Open ``nbx_user.py`` file and you will see ``get_resource`` function that can used to modify your Pod requirements. Your Job
starts with reasonable default for most of the workloads.

**How do I transfer file to an object store like S3?**

We are working on ``nbox.Relic`` that will act as a single panel for multiple object stores like AWS S3, GCP Bukets, etc.

**How to run jobs at schedule?**

Open ``nbx_user.py`` file and you will see ``get_schedule`` function that can used to modify schedule for your Job. Note that
this is not used when uploading a serving.

**How can I run distributed computing?**

tl;dr WIP, will be released along with NBX-Relics.

This is a super important question. It is easy to have a lot of false positives when it comes to distributed computing. When
user writes a script (or any piece of code) what they are expecting is that it will run across a bunch of machines and get me
back the results where ever the query is being made from. It's hard but we are working on it.

**How can I run a notebook with Jobs?**

There is built in support for running jupyter notebooks using `nbox.lib.notebook.NotebookRunner` Operator. This takes in the
path of the notebook and it will run it on the cloud, and that is the API contract. Now the underlying mechanisms to do so
will keep on changing as we keep upgrading our technology. Example usage:

.. code-block:: python

  filepath = "trainer_v12_2.ipynb"
  op = NotebookRunner(filepath)

