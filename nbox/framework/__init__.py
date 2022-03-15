r"""This submodule concerns itself with conversion of different framworks to other frameworks.
It achieves this by providing a fix set of functions for each framework. There are a couple of
caveats that the developer must know about.

1. We use joblib to serialize the model, see `reason <https://stackoverflow.com/questions/12615525/what-are-the-different-use-cases-of-joblib-versus-pickle>`_ \
so when you will try to unpickle the model ``pickle`` will not work correctly and will throw the error
``_pickle.UnpicklingError: invalid load key, '\x00'``. So ensure that you use ``joblib``.

2. Serializing torch models directly is a bit tricky and weird, you can read more about it
`here <https://github.com/pytorch/pytorch/blob/master/torch/csrc/jit/docs/serialization.md>`_,
so technically pytorch torch.save() automatically pickles the object along with the required
datapoint (model hierarchy, constants, data, etc.)

Lazy Loading
------------

All the dependencies are checked at runtime only, meaning all the modules coded can be referenced
removing blockers and custom duct taping.

Documentation
-------------
"""

from .on_ml import  *
from .on_operators import *
from .model_spec_pb2 import *

try:
  from .ml import *
except ImportError:
  from .autogen import compile
  compile()
  from .ml import *


def get_model_functions(py_model):
  try:
    __import__(Framework_torch._load_framework)
    if Framework_torch._conditional(py_model):
      methods = Framework_torch._METHODS
      return methods
  except ImportError:
    pass
  return {}

