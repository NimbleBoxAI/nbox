.. nbox documentation master file, created by
   sphinx-quickstart on Sat Oct  9 20:06:49 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to nbox's documentation!
================================

Hi, there 👾!

NimbleBox.ai started to reduce entry barrier to production grade AI workloads for developers. We started that by introducing our `platform <https://docs.nimblebox.ai/>`_\ , and now with our SDK nbox. Inference of models is the most common operation we see on our platform, and is a bottleneck for non-ML developers who just want to use that as a function and don't want to setup a pipeline.

Find the complete open source code on `github <https://github.com/NimbleBoxAI/nbox>`_. Install the package by running the command:

.. code-block::

   pip install nbox

..

   In order to effectively use this package, you must have a password set. You can get it by going to Profile → Reset Password.


When loading ``nbox`` for the first time, it will prompt you the username and password and create a secrets file at ``~/.nbx/secrets.json``. This file then contains all the information that you don't have to fetch manually again.

``nbox`` is a new package designed ground up with inference and production grade deployment in mind. The input to model is called input_object and it can be a string, array-like, binary-like in form of a list or Dict. Read full technical document `here <#>`_.

.. code-block::

   import nbox

   # As all these models come from the popular frameworks you use such as 
   # torchvision, efficient_pytorch or hf.transformers
   model = nbox.load("torchvision/mobilenetv2", pretrained = True)

   # nbox makes inference the priority so you can
   out = model('cat.jpg')                          # pass it image path
   out = model(np.array(Image.open('cat.jpg')))    # pass it numpy arrays
   out = model(['cat.jpg', Image.open('cat.jpg')]) # pass it a list for batch inference

``nbox`` can load any underlying model from package and can consume anything (eg. code above) whether it is ``filepaths``\ , ``PIL.Image`` or just a ``numpy array``.


Indices and tables
==================

.. toctree::
   :maxdepth: 2
   :caption: Contents

   modules
   nbox.load
   nbox.model
   nbox.framework
   nbox.network
   nbox.utils
   nbox.user
   

* :ref:`genindex`
* :ref:`modindex`
