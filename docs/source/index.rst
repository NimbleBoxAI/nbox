.. nbox documentation master file, created by
   sphinx-quickstart on Sat Oct  9 20:06:49 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to nbox's documentation!
================================

..

   Good documentation is always a work in silence.

Hi, there 👾!

``nbox`` is SDK for `NimbleBox.ai <https://app.nimblebox.ai/>`_\ , it provides built in access to
all the APIs and packages them in the most user friendly manner. Writing MLOps pipelines from scratch
can be a daunting task and this is a breakdown of how nbox works. Find the complete open source code
on `github <https://github.com/NimbleBoxAI/nbox>`_. Install the package from ``pipy``:

.. code-block::

   pip install nbox


For convinience you should add ``nbox`` to your path by setting up an alias. Throughout the rest of the
documentation we will be using ``nbx`` as the CLI:

.. code-block::

   # go to your .bashrc or .zshrc and add
   alias nbx="python3 -m nbox"

..

   In order to effectively use this package, you must have a password set.
   You can get it by going to Profile → Reset Password.

When loading ``nbox`` for the first time, it will prompt you the username and password and create a secrets
file at ``~/.nbx/secrets.json``. This file then contains all the information that you don't have to
fetch manually again.

APIs
----

The objective is to make using ML 🥶. For this it is paramount that APIs be deeper, user functions be
kept to minimum and most relavant. This documentation contains the full spec of everything, but here's
all the APIs you need to know:

.. code-block::

   nbox
   ├── Model          # Framework agnostic Model
   │   ├── __call__
   │   ├── deploy
   │   ├── train_on_instance (WIP)
   │   └── train_on_jobs (WIP)
   ├── Operators      # How jobs are combinations of operators
   │   ├── __call__
   │   └── deploy
   ├── Jobs           # For controlling all your jobs
   │   ├── logs       # stream logs right on your terminal
   │   └── trigger    # manually trigger a job
   └── Instance
      ├── __call__    # Run any command on the instance
      └── mv (WIP)    # Move files to and from NBX-Build

Though the underlying framework will keep on increasing we already use Protobufs, gRPC along with auto
generating code files.

CLI
---

To provide zero differences between using CLI and packages, we use python-fire that makes CLIs using
python objects. Example, let's just say you want to turn off any instance

.. code-block::

   # In case of script
   Instance(i = "nbox-dev", workspace_id = "99mhf3h").stop()

   # In case of CLI
   nbx build --i="nbox-dev" --workspace_id="99mhf3h" stop


SSH
"""

Or you can directly SSH into instances (Read more aout).

.. code-block::

   nbx tunnel 8000 --i="nbox-dev"

GET
"""

Or you can see the status by making GET calls from CLI along with ``jq``:

.. code-block::

   $ nbx get "workspace/99mhf3h/projects/2892" | tail -n 1 | jq
   > {
      "data": {
         "auto_backup": null,
         "auto_shutdown_time": -1,
         "autoshutdown": false,
         "clone_access": false,
         "created_time": "1647858175.0",
         "creator": "Bruce Wayne",
         "dedicated_hw": false,
         "editor": "csv2",
         ...


If you want to see something be added or found bug, `raise an issue <https://github.com/NimbleBoxAI/nbox/issues/new>`_.

Hope you enjoy this.

**Yash Bonde** (NimbleBox.ai R&D)


Index
=====

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   quick.0
   quick.jobs
   nbox.cli
   nbox.nbxlib.ops

.. toctree::
   :maxdepth: 2
   :caption: Commonly used APIs

   nbox.model
   nbox.operator
   nbox.jobs
   nbox.instance
   nbox.framework.ml
   

.. toctree::
   :maxdepth: 2
   :caption: R&D

   nbox.subway
   nbox.framework.rst
   

.. toctree::
   :maxdepth: 2
   :caption: Engineering

   nbox.messages
   nbox.sub_utils.ssh
   nbox.auth.rst


* :ref:`genindex`
* :ref:`modindex`
