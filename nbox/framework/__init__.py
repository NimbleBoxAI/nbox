"""
This is the hidden closet with all messy and cool things.

This submodule has files for managing the adapters for different frameworks. The important
repositories with the logic start with ``on_``, why?

#. ``on_ml``: contains the code for exporting and sharing the weights across different\
    ml libraries like pytorch, tensorflow, haiku, etc.
#. ``on_operators``: contains the code for exporting and importing operators from frameworks\
    like Airflow, Prefect, etc.
#. ``on_functions``: This library contains the code for static parsing of python and\
    creating the NBX-JobFlow

Then there is one main file for auto generating the code

#. ``autogen``: Which contains the compilers for different code generations like ``ml.py``,\
    there will be more in the future.

Then there are code that is generated:

#. ``ml``: This is the code for all ml frameworks, their exporting and importing using\
    message-stub format. That is all the user arguments became a dataclass which user has\
    to implement according to their requirments.
#. ``*_pb2.py/pyi``: These are the protobuf stubs generated, as of this writing there is only\
    one proto called ``ModelSpec`` which is like ``JobProto`` but for models.

The ``protos/`` folder contains all the proto definitions. Some of the files in here do not
have documentation and that is intentionally to hide all the complexities form the user.
If you are interested, you can read the source code directly from Github.

"""

# if you like what you see and want to work on more things like this, reach out research@nimblebox.ai

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
  """Try to infer the functions from the model"""
  try:
    __import__(Framework_torch._load_framework)
    if Framework_torch._conditional(py_model):
      methods = Framework_torch._METHODS
      return methods
  except ImportError:
    pass
  return {}
