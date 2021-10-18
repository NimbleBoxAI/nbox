nbox Model
==========

``nbox.Model`` is a simple wrapper on top of your ``pytorch`` and ``scikit-learn`` models that simplifies
using the model based functionalities and products from NimbleBox.ai dead-simple to use. These include
deploying models onto a fully autoscaling kubernetes pods, exporting models between multiple different
formats or compiling them to high performance `OVMS <https://docs.openvinotoolkit.org/latest/>`_ containers.

Some things to understand:

1. **deployment_type:** This is the server type to be used, there are two types, a power serving ``ovms2``
(written C++) and a flexible ``nbox`` (written in python).

2. **runtime:** This is the serialisation for running the model, there are three different types based on the
serving method (``deployment_type``):

* When using ``deployment_type=='nbox'``:
    * ``onnx`` this uses the ONNX Runtime, runs with GPUs and CPUs
    * ``torchscript`` this is exclusively for ``pytorch`` models, runs with GPUs and CPUs
* When using ``deployment_type=='ovms2'``, the runtime is automatically set

This table can be a handy guide in understanding what is happening.


.. list-table::
   :widths: 35 35 25 25
   :header-rows: 1

   * - ``runtime``
     - ``deployment_type``
     - Pytorch ``pt``
     - Scikit ``sk``
   * - ``onnx``
     - ``ovms2``
     - ✅
     - ❌
   * - ``onnx``
     - ``nbox``
     - ✅
     - ✅
   * - ``torchscript``
     - ``nbox``
     - ✅
     - 🧩 NA
   * - ``pkl`` "pickle"
     - ``nbox``
     - use ``torchscript``
     - 🚧 in progress


Documentation
-------------


.. automodule:: nbox.model
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __call__
