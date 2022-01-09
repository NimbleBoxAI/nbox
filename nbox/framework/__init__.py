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


__all__ = ["get_meta"]

from types import SimpleNamespace
from ..utils import _isthere, folder, join

def update_all_lazy_loading(*modules, fname):
    """Lazy load modules and update the ``__all__`` list"""
    import sys
    global __all__
    if _isthere(*modules):
        sys.path.append(join(folder(__file__), f"{fname}.py"))
        _all_name = f"{fname}_all"
        eval(f"from .{fname} import *")
        eval(f"from .{fname} import __all__ as {_all_name}")
        maps = {f"{x}": globals()[x] for x in _all_name}
        maps.update({"IMPORTS": [*modules]})
        _name_of_module = fname.strip("_")
        eval(f"{_name_of_module} = SimpleNamespace(**maps)")
        __all__ += [f"{_name_of_module}"]

# update_all_lazy_loading("torch", fname = "__pytorch")  
# update_all_lazy_loading("sklearn", "sk2onnx", fname = "__sklearn")


_pt_modules = ["torch"]
if _isthere(*_pt_modules):
    from .__pytorch import *
    from .__pytorch import __all__ as _pt_all
    maps = {f"{x}": globals()[x] for x in _pt_all}
    maps.update({"IMPORTS": _pt_modules})
    pytorch = SimpleNamespace(**maps)
    __all__ += ["pytorch"]

_sk_modules = ["sklearn", "sk2onnx"]
if _isthere(*_sk_modules):
    from .__sklearn import *
    from .__sklearn import __all__ as _sk_all
    maps = {f"{x}": globals()[x] for x in _sk_all}
    maps.update({"IMPORTS": _sk_modules})
    sklearn = SimpleNamespace(**maps)
    __all__ += ["sklearn"]

_onnx_modules = ["onnx", "onnxruntime"]
if _isthere(*_onnx_modules):
    maps = {"IMPORTS": _onnx_modules}
    onnx = SimpleNamespace(**maps)
    __all__ += ["onnx"]
