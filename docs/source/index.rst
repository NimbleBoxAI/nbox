.. nbox documentation master file, created by
   sphinx-quickstart on Sat Oct  9 20:06:49 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to nbox's documentation!
================================

..

   Good documentation is always a work in silence and progress.

Hi, there 👾!

NimbleBox.ai started to reduce entry barrier to production grade AI workloads for developers. We started that by introducing our `platform <https://docs.nimblebox.ai/>`_\ , and now with our SDK nbox. Inference of models is the most common operation we see on our platform, and is a bottleneck for non-ML developers who just want to use that as a function and don't want to setup a pipeline.

Find the complete open source code on `github <https://github.com/NimbleBoxAI/nbox>`_. Install the package by running the command:

.. code-block::

   pip install nbox

..

   In order to effectively use this package, you must have a password set. You can get it by going to Profile → Reset Password.

When loading ``nbox`` for the first time, it will prompt you the username and password and create a secrets file at ``~/.nbx/secrets.json``. This file then contains all the information that you don't have to fetch manually again.

``nbox`` is a new package designed ground up with inference and production grade deployment in mind. The input to model is called input_object and it can be a string, array-like, binary-like in form of a list or Dict.

.. code-block:: python

   import nbox

   # As all these models come from the popular frameworks you use such as 
   # torchvision, efficient_pytorch or hf.transformers
   model = nbox.load("torchvision/mobilenetv2", pretrained = True)

   # nbox makes inference the priority so you can feed strings to cats?
   out = model("cat.jpg")
   
   # deploy model on NibleBox.ai Kubernetes cluster
   url, key = model.deploy("cat.jpg", wait_for_deployment = True)

   # load a cloud model like WTH
   model = nbox.load(url, key)

   # use as if locally
   out = model("cat.jpg")

``nbox`` can load any underlying model from package and can consume anything (eg. code above) whether it is ``filepaths``\ , ``PIL.Image`` or just a ``numpy array``.

The objective is to make using ML models 🥶. For this it is paramount that the number of lines of code
be very small. If it is very small, a smart engineer can open it up and debug it. The metric to always
aim for is total lines of code which should stand below 1000 for ``top_dir``, other submodules can
contain ``dicts`` that occupy a large % of code.

As of ``nbox == 0.8.1`` this is where the stats stand!

.. code-block::

   SLOC	Directory	SLOC-by-Language (Sorted)
   1181    top_dir         python=1181
   88      framework       python=88

   Totals grouped by language (dominant language first):
   python:        1269 (100.00%)


If you want to see something be added or found bug, `raise an issue <https://github.com/NimbleBoxAI/nbox/issues/new>`_.


Hope you enjoy this.

**Yash Bonde** (NimbleBox.ai R&D)




Index
=====

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   quick
   quick_deploy
   quick_plug
   quick_skl
   nbox.cli
   nbox-serving

.. toctree::
   :maxdepth: 2
   :caption: Documentations

   nbox.load
   nbox.model
   nbox.parsers
   nbox.framework
   nbox.network
   nbox.utils
   nbox.auth

.. toctree::
   :maxdepth: 2
   :caption: Operators

   nbox.operators
   nbox.operators.operator
   nbox.operators.lib

.. toctree::
   :maxdepth: 2
   :caption: Jobs

   nbox.jobs
   nbox.jobs.utils


.. toctree::
   :maxdepth: 2
   :caption: R&D

   in_progress


* :ref:`genindex`
* :ref:`modindex`
