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

Lazy loading is a mechanism that allows you to load the model only when you need it, this is easier said than
done because you need to add many checks and balances at varaious locations in the code. The way it works here
is that we check from a list of modules which are required to import which part of the model.


Documentation
-------------
"""

from .common import IllegalFormatError
from ..utils import _isthere

# this function is for getting the meta data and is framework agnostic, so adding this in the
# __init__ of framework submodule
def get_meta(input_names, input_shapes, args, output_names, output_shapes, outputs):
  """Generic method to convert the inputs to get ``nbox_meta['metadata']`` dictionary"""
  # get the meta object
  def __get_struct(names_, shapes_, tensors_):
    return {
      name: {
        "dtype": str(tensor.dtype),
        "tensorShape": {"dim": [{"name": "", "size": x} for x in shapes], "unknownRank": False},
        "name": name,
      }
      for name, shapes, tensor in zip(names_, shapes_, tensors_)
    }

  meta = {"inputs": __get_struct(input_names, input_shapes, args), "outputs": __get_struct(output_names, output_shapes, outputs)}

  return meta


from .__nbx import NBXDeployMixin

all_mixin = [NBXDeployMixin]

if _isthere("torch"):
  from .__pytorch import TorchMixin
  all_mixin.append(TorchMixin)

if _isthere("sklearn", "sk2onnx"):
  from .__sklearn import SklearnMixin
  all_mixin.append(SklearnMixin)

if _isthere("onnxruntime"):
  from .__onnxrt import ONNXRtMixin
  all_mixin.append(ONNXRtMixin)

def get_mixin(model):
  for m in all_mixin:
    try: return m.load_model(model)
    except IllegalFormatError: pass

  raise IllegalFormatError(f"This model: '{type(model)}' is not supported")

__all__ = [
  "get_meta",
  "get_mixin",
]