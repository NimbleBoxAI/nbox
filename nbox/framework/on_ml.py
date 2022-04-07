"""
I started with the initial assumption that abstracting away the details of
processing the data was doable but unfortunately I was not able to get it to
work reliably. We redid the entire thing by manually defining the code and
operations but it was still super hard. So the answer was to generate code
that the user will have to implement.

It loosely works like this, let's just say I am trying to write code for
``torch_to_onnx`` module. ``ModelSpec`` expects that ``source`` and ``target``
contain all the required information for running this model in another
location. Now the actual code is only of two lines, but the many different
frameworks, with their different functions and different inputs means that
any scope of manual process is gone out the window. So we basically create a
registry that can store all the information and pass it along to ``nbox.Model``.

The implementation of the registry is a ``@register`` decorator, which takes
in all the user information and underneath structures the entire thing and
stores it, if needed to be written down for compilation.

Read the code for best understanding.
"""

import os

from .. import utils as U
from ..utils import isthere, logger
from .autogen import ml_register
from .model_spec_pb2 import ModelSpec, Tensor

class NboxOptions:
  def __init__(self, model_name: str = None, folder: str = None, create_folder = False):
    model_name = model_name or U.get_random_name().replace("-", "_")
    if create_folder:
      self.folder = U.join(folder if folder != None else U.NBOX_HOME_DIR, model_name)
      self.model_name = model_name
      os.makedirs(self.folder, exist_ok = False)
    else:
      self.folder = folder
      self.model_name = model_name

@ml_register.conditional(framework = "torch",)
def torch_condition(model):
  import torch
  if isinstance(model, torch.nn.Module):
    return True
  return False

@ml_register.register(
  framework = "torch",
  target = "torch",
  stub_name = "torch_to_torchscript",
  message_name = "TorchToTorchscript",
  target_processor_name = "torchscript_to_torch",
  export_fn_import = "from torch.jit import trace",
  dependencies = ["-f https://download.pytorch.org/whl/cpu/torch_stable.html", "torch"],
)
def torch_export_torchscript(user_options, nbox_options, spec: ModelSpec):
  import torch
  from torch.jit import trace
  traced_model = trace(**user_options.__dict__)
  filepath = U.join(nbox_options.folder, "model.pt")
  spec.source.path = filepath
  spec.inputs.extend([
    Tensor(
      name = str(i),
      shape = tuple(t.shape),
      dtype = str(t.numpy().dtype)
    ) for i, t in enumerate(user_options.example_inputs)
  ])

  # actual export
  torch.jit.save(traced_model, filepath)
  return spec

@ml_register.register(
  framework = "torch",
  target = "torch",
  stub_name = "torchscript_to_torch",
  message_name = "TorchscriptToTorch",
  export_fn_import = "from torch.jit import load",
  dependencies = ["-f https://download.pytorch.org/whl/cpu/torch_stable.html", "torch"],
  ignore_args= ["f"],
)
def torch_import_torchscript(user_options, nbox_options, spec: ModelSpec):
  from torch.jit import load
  model = load(U.join(nbox_options.folder, "model.pt"), map_location="cpu")
  return model

@ml_register.register(
  framework = "torch",
  target = "onnx",
  target_processor_name = "onnx_runtime",
  stub_name = "torch_to_onnx",
  message_name="TorchToOnnx",
  export_fn_import="from torch.onnx import export",
  dependencies=["-f https://download.pytorch.org/whl/cpu/torch_stable.html", "torch"],
  ignore_args = ["f"],
)
def torch_export_onnx(user_options, nbox_options, spec: ModelSpec):
  import torch
  from torch.onnx import export
  user_options.f = nbox_options.filepath # override the folder
  inputs = user_options.args
  if isinstance(inputs, torch.Tensor):
    spec.inputs.append(Tensor(name = '0', shape = tuple(inputs.shape), dtype = str(inputs.numpy().dtype)))
  elif isinstance(inputs, tuple):
    if any(isinstance(x, dict) for x in inputs):
      if not inputs[-1] == {}:
        logger.warning(f"Last input is not an empty dictionary and is recommended by ONNX Export")
    for i,t in enumerate(inputs):
      if isinstance(t, torch.Tensor):
        spec.inputs.append(Tensor(name = str(i), shape = tuple(t.shape), dtype = str(t.numpy().dtype)))
      elif isinstance(t, dict):
        for k,v in t.items():
          spec.inputs.append(Tensor(name = k, shape = tuple(v.shape), dtype = str(v.numpy().dtype)))

  # actual export
  export(**user_options.__dict__)
  return spec


################################################################################
# User Guide
# ==========
# This 
################################################################################


@isthere("torch")
def _get_torch_model():
  import torch
  class Feedforward(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
      super(Feedforward, self).__init__()
      self.input_size = input_size
      self.hidden_size = hidden_size
      self.network = torch.nn.Sequential(
        torch.nn.Linear(self.input_size, self.hidden_size),
        torch.nn.Softmax(-1),
        torch.nn.Linear(self.hidden_size, 1),
        torch.nn.Sigmoid(),
      )

    def forward(self, x):
      output = self.network(x)
      return output

  import random

  _dim = random.choice(range(10))
  return (
    {
      "model": Feedforward(input_size = _dim, hidden_size = 2)
    }, # nbox.Model kwargs
    torch.randn(1, _dim), # example input
  )

@isthere("sklearn", "numpy")
def _get_sklearn_model():
  from sklearn import linear_model
  reg = linear_model.LinearRegression()
  data = [[0, 0], [1, 1], [2, 2]], [0, 1, 2]

  return (
    {
      "model": reg,
      "method": "fit",
    }, # nbox.Model kwargs
    data, # example input
  )

def _get_default_forward():
  class SomeCaller:
    def __init__(self):
      pass

    def __call__(self):
      print("Add your logic here")

  return (
    {
      "model": SomeCaller(),
    }, # nbox.Model kwargs
    None, # example input
  )




class InvalidProtocolError(Exception):
  pass

# class ModelOutput(U.DBase):
#   __slots__ = [
#     "inputs", # :Any
#     "outputs", # :Any
#   ]

#   def serialise():
#     raise NotImplementedError

# # class ModelSpec(DBase):
# #   __slots__ = [
# #     # where from
# #     "src_framework", # :str: name of the source framework
# #     "src_framework_version", # :str
# #     'export_path', # str: there is no reason for the Pod to know anything about the user

# #     # where to
# #     "export_type", # :str
# #     "exported_time", # :str: UTC time when the model was exported

# #     # how to
# #     "load_method", # :str: The classmethod to call to load this model
# #     "load_kwargs", # :dict: kwargs to pass to the load method
# #     "io_dict", # :dict: obtained from above function

# #     # 'what to' is the serving script!
# #     "required_packages", # :list: list of things to write in requirements.txt
# #   ]

# class FrameworkAgnosticProtocol(object):
#   """
#   This Protocol only aims to serve as an abstracted layer that calls the underlying framework
#   specific code and returns the output. This mainly is intended to implement three functions:
#   * forward: the forward pass of the model
#   * export: that takes in a string for export format and arguments for it
#   * deserialise: that takes the model spec object and returns the loader for this model
#   https://nimblebox.notion.site/nbox-FrameworkAgnosticProtocol-6b39249316b1497b8ad9ff8f02b227f0
#   """
#   # def __init__(self, i0: Any, i1: Any) -> None

#   def forward(self, input_object) -> ModelOutput:
#     out = self._model(**input_object)
#     return ModelOutput(inputs = input_object, outputs = out)

#   # def export(self, format: str, input_object: Any, export_model_path: str, **kwargs) -> ModelSpec:
#   #   raise InvalidProtocolError(f"export method not implemented: {self.__class__.__name__} -> {format}")

#   def serialise(self, format):
#     if format not in self.serialisations:
#       raise InvalidProtocolError(f"Format {format} not supported for torch export")
#     return self.serialisations[format]

#   @staticmethod
#   def deserialise(model_meta: ModelSpec) -> Tuple[Any, Any]:
#     raise InvalidProtocolError(f"deserialise method not implemented: '{model_meta.export_type}'")


# def get_io_dict(input_object, call_fn, forward_fn):
#   """Generates and returns an io_dict by performing forward pass through the model

#   Args:
#       input_object (Any): Input to a model
#       call_fn (Callable): function that Model.model employs to do a forward pass.
#       forward_fn (Callable): forward() function of the Model

#   Returns:
#       io : io_dict
#   """
#   logger.info(f"Generating io_dict")
#   out = forward_fn(input_object)
#   args = inspect.getfullargspec(call_fn)
#   args.args.remove("self")

#   # Create the i/o dict
#   def __get_struct(object):

#     def process_dict(x, curr_idx) -> dict:
#       #Process Dictionaries
#       results = {}
#       for key in x.keys():
#         parsed, curr_idx = parse(x[key], name=key, curr_idx=curr_idx)
#         results[key] = parsed
#       return results, curr_idx

#     def process_container(x, curr_idx) -> dict:
#       #Handles lists, sets and tuples
#       results = []
#       for element in x:
#         parsed, curr_idx = parse(element, None, curr_idx)
#         results.append(parsed)

#       return results, curr_idx

#     def parse(x, name=None, curr_idx=0) -> dict:
#       # Parses objects to generate iodict
#       dtype = None
#       if name is None:
#         name = f"tensor{curr_idx}"
#         curr_idx += 1

#       if hasattr(x, "dtype"):
#         dtype = str(x.dtype)

#       if isinstance(x, dict):
#         return process_dict(x, curr_idx)

#       elif isinstance(x, (list, set, tuple)):
#         return process_container(x, curr_idx)

#       elif hasattr(x, "shape"):
#         dim_names=[""]*len(x.shape)
#         if hasattr(x, "names"):
#           dim_names=x.names
#         return {"name": name, "dtype": dtype, "tensorShape": {"dim":[{'name':dim_names[y], "size":x.shape[y]} for y in range(len(x.shape))],"unknownRank":False}}, curr_idx
#       else:
#         return {"name": name, "dtype": dtype, "shape": None}, curr_idx

#     return parse(object)[0]

#   io = {
#     "inputs": __get_struct(out.inputs),
#     "outputs": __get_struct(out.outputs),
#     "arg_spec": {
#       "args": args.args,
#       "varargs": args.varargs,
#       "varkw": args.varkw,
#       "defaults": args.defaults,
#       "kwonlyargs": args.kwonlyargs,
#       "kwonlydefaults": args.kwonlydefaults,
#     }
#   }
#   return io


# ################################################################################
# # NimbleBox.ai Deployments
# # ========================
# # NBX-Deploy is a service that you can use to load the API endpoints where it is
# # deployed. This is a special kind of method because it can consume all the
# # objects that other models can consume.
# ################################################################################

# # TODO:@yashbonde
# # class NBXModel(FrameworkAgnosticProtocol):
# #   @U.isthere("numpy", soft=False)
# #   def __init__(self, url_key: Tuple[str, str]):
# #     logger.debug(f"Trying to load as NBX-Api")
# #     if not (isinstance(url_key, (tuple, list)) and len(url_key) == 2):
# #       raise InvalidProtocolError(f"Invalid url_key for NBX-Api")

    

# #     self.url = url
# #     self.key = key

# #     # fetch the metadata from the cloud
# #     model_url = url.rstrip("/")
# #     logger.debug("Getting model metadata")
# #     URL = secret.get("nbx_url")
# #     r = requests.get(f"{URL}/api/model/get_model_meta", params=f"url={model_url}&key={key}")
# #     try:
# #       r.raise_for_status()
# #     except:
# #       raise ValueError(f"Could not fetch metadata, please check status: {r.status_code}")

# #     # start getting the metadata, note that we have completely dropped using OVMS meta and instead use nbox_meta
# #     content = json.loads(r.content.decode())["meta"]
# #     nbox_meta = content["nbox_meta"]

# #     all_inputs = nbox_meta["metadata"]["inputs"]
# #     templates = {}
# #     for node, meta in all_inputs.items():
# #       templates[node] = [int(x["size"]) for x in meta["tensorShape"]["dim"]]
# #     logger.debug("Cloud infer metadata obtained")

# #     category = nbox_meta["spec"]["category"]

# #     # if category is "text" or if it is dict then any key is "text"
# #     tokenizer = None
# #     max_len = None
# #     if category == "text" or (isinstance(category, dict) and any([x == "text" for x in category.values()])):
# #       import transformers

# #       model_key = nbox_meta["spec"]["model_key"].split("::")[0].split("transformers/")[-1]
# #       tokenizer = transformers.AutoTokenizer.from_pretrained(model_key)
# #       max_len = templates["input_ids"][-1]

# #     image_parser = ImageParser(cloud_infer=True, post_proc_fn=lambda x: x.tolist(), templates=templates)
# #     text_parser = TextParser(tokenizer=tokenizer, max_len=max_len, post_proc_fn=lambda x: x.tolist())

# #   def forward(self, model_input):
# #     import numpy as np

# #     logger.debug(f"Hitting API: {self.model_or_model_url}")
# #     st = time()
# #     # OVMS has :predict endpoint and nbox has /predict
# #     _p = "/" if "export_type" in self.nbox_meta["spec"] else ":"
# #     json = {"inputs": model_input}
# #     if "export_type" in self.nbox_meta["spec"]:
# #       json["method"] = method
# #     r = requests.post(self.url + f"/{_p}predict", json=json, headers={"NBX-KEY": self.key})
# #     et = time() - st
# #     out = None

# #     try:
# #       r.raise_for_status()
# #       out = r.json()

# #       # first try outputs is a key and we can just get the structure from the list
# #       if isinstance(out["outputs"], dict):
# #         out = {k: np.array(v) for k, v in r.json()["outputs"].items()}
# #       elif isinstance(out["outputs"], list):
# #         out = np.array(out["outputs"])
# #       else:
# #         raise ValueError(f"Outputs must be a dict or list, got {type(out['outputs'])}")
# #       logger.debug(f"Took {et:.3f} seconds!")
# #     except Exception as e:
# #       logger.debug(f"Failed: {str(e)} | {r.content.decode()}")
# #       raise e


# ################################################################################
# # Torch
# # =====
# # Torch is a deep learning framework from Facebook. It is aimed towards a simple
# # interface and is a huge inspiration to me. User needs to define the
# # config/init and forward pass. This is the minimum input you need to take from
# # a user to get all the output
# ################################################################################




# #   @staticmethod
# #   def deserialise(model_meta: ModelSpec) -> Tuple[Any, Any]:
# #     import torch
# #     logger.debug(f"Deserialising torch model from {model_meta.export_path}")

# #     kwargs = model_meta.load_kwargs

# #     if model_meta.export_type == "torchscript":
# #       lp = kwargs.pop("logic_path")
# #       logger.debug(f"Loading logic from {lp}")
# #       model = torch.jit.load(model_meta.load_kwargs["model"], map_location=kwargs["map_location"])
# #       return model
# #     else:
# #       raise InvalidProtocolError(f"Unknown format: {model_meta.export_type}")


# ################################################################################
# # ONNX Runtime
# # ============
# # OnnxRuntime is a C++ library that allows you to run ONNX models in your
# # application. It is a wrapper around the ONNX API. It *does not*
# # work on Apple M1 machines and so suck it up! But this is useful for server
# # side processing.
# ################################################################################

# class ONNXRtModel(FrameworkAgnosticProtocol):
#   @U.isthere("onnx", "onnxruntime", "numpy", soft = False)
#   def __init__(self, m0):
#     import onnx
#     import onnxruntime as ort
#     logger.debug(f"Trying to load as ONNX Runtime")

#     if not isinstance(m0, onnx.onnx_ml_pb2.ModelProto):
#       raise InvalidProtocolError(f"First input must be a ONNX model, got: {type(m0)}")

#     self._sess = ort.InferenceSession(self._model.SerializeToString())

#   def forward(self, input_object) -> ModelOutput:
#     # import numpy as np
#     # model_inputs = self._logic(input_object)
#     # input_names = [i.name for i in self._sess.get_inputs()]
#     # output_names = [o.name for o in self._sess.get_outputs()]
#     # if isinstance(model_inputs,dict):
#     #   tensor_dict = model_inputs
#     #   if isinstance(list(model_inputs.values())[0],SklearnInput):
#     #     tensor_dict = {list(model_inputs.keys())[0]:list(model_inputs.values())[0].inputs}
#     #   model_inputs = {name: tensor for name, tensor in zip(input_names, tuple(tensor_dict.values()))}
#     # else:
#     #   raise ValueError("Your Logic needs to return dictionary")

#     # # conversion to proper arrays -> if can be numpy-ed and float32
#     # for key, value in model_inputs.items():
#     #   if hasattr(value, "numpy"):
#     #     value = value.numpy()
#     #   if value.dtype in (np.float64, np.float32, np.float16):
#     #     value = value.astype(np.float32)
#     #   elif value.dtype in (np.int64, np.int32, np.int16, np.int8, np.int64):
#     #     value = value.astype(np.int32)
#     #   model_inputs[key] = value

#     # res = self._sess.run(output_names=output_names, input_feed=model_inputs)

#     res = self._sess.run(**input_object)
#     return ModelOutput(
#       inputs = input_object,
#       outputs = res,
#     )

#   @staticmethod
#   def deserialise(model_meta: ModelSpec) -> Any:
#     import onnx
#     logger.debug(f"Deserialising ONNX model from {model_meta.export_path}")

#     if model_meta.export_type == "onnx":
#       model = onnx.load(model_meta.load_kwargs["model"])
#       return model
#     else:
#       raise InvalidProtocolError(f"Unknown format: {model_meta.export_type}")


# ################################################################################
# # Sklearn
# # =======
# # Scikit-learn is a Python library that allows you to use classical machine
# # learning algorithms in your application. Its versatility is huge and that
# # means that a dedicate Input DataModel is required.
# ################################################################################

# class SklearnModel(FrameworkAgnosticProtocol):
#   @U.isthere("sklearn", "numpy", soft = False)
#   def __init__(self, m1):
#     logger.debug(f"Trying to load as Sklearn")
#     if not "sklearn" in str(type(m1)):
#       raise InvalidProtocolError

#     self._model = m1

#     self.exporters = {
#       "pkl": U.to_pickle
#     }
#     self.loaders = {
#       "pkl": U.from_pickle
#     }

#   def forward(self, input_object: Dict) -> ModelOutput:
#     """Sklearn covers a far wider range of models and is therefore more complicated to cover.
#     For now the input is expected to be a Dict that has "inputs" dict that will be fed to the model.
#     And it has "method" key that will be called upon the model i.e. whether user wants
#     ``predict`` (default) or ``predict_proba`` is left upto the user.
#     """

#     # if "sklearn.neighbors.NearestNeighbors" in str(type(self._model)):
#     #   method = getattr(self._model, "kneighbors") if method == None else getattr(self._model, method)
#     # elif "sklearn.cluster" in str(type(self._model)):
#     #   if any(
#     #     x in str(type(self._model)) for x in ["AgglomerativeClustering", "DBSCAN", "OPTICS", "SpectralClustering"]
#     #   ):
#     #     method = getattr(self._model, "fit_predict")
#     # else:
#     #   try:
#     #     method = getattr(self._model, "predict") if method == None else getattr(self._model, method)
#     #   except Exception as e:
#     #     logger.debug(f"[ERROR] Model Prediction Function is not yet registered {e}")

#     model_input = input_object.get("inputs", None)
#     method = input_object.get("method", "predict")
#     method_fn = getattr(self._model, method, None)
#     if method_fn == None:
#       logger.error(f"Method {method} not found in SklearModel")
#       raise ValueError(f"Method {method} not found in model")
#     out = method(**model_input)

#     return ModelOutput(inputs = model_input, outputs = out,)

#   @U.isthere("skl2onnx", soft = False)
#   def export_to_onnx(self, input_object, export_model_path, model_file_name="model.onnx", opset_version=None,) -> ModelSpec:
#     import skl2onnx
#     import skl2onnx.common.data_types as dt

#     __NP_DTYPE_TO_SKL_DTYPE = {
#       "bool_": dt.BooleanTensorType,
#       "double": dt.DoubleTensorType,
#       "int16": dt.Int16TensorType,
#       "complex128": dt.Complex128TensorType,
#       "int32": dt.Int32TensorType,
#       "uint16": dt.UInt16TensorType,
#       "complex64": dt.Complex64TensorType,
#       "float16": dt.Float16TensorType,
#       "int64": dt.Int64TensorType,
#       "str_": dt.StringTensorType,
#       "uint32": dt.UInt32TensorType,
#       "float64": dt.FloatTensorType,
#       "uint64": dt.UInt64TensorType,
#       "int8": dt.Int8TensorType,
#       "uint8": dt.UInt8TensorType,
#     }

#     export_path = join(export_model_path, model_file_name)
#     method = input_object.get("method", "predict")
#     method = getattr(self._model, method, None)
#     if method == None:
#       logger.error(f"Method {method} not found in SklearModel")
#       raise ValueError(f"Method {method} not found in model")
#     iod = get_io_dict(input_object, method, self.forward)

#     # the args need to be converted to proper formatted data types
#     initial_types = []
#     input_names, input_shapes, input_dtypes = [],[],[]

#     # Case for input object with single tensor
#     inputs=iod["inputs"]
#     if isinstance(inputs['name'], str):
#       input_names.append(inputs["name"])
#       input_dtypes.append(inputs["dtype"])
#       tensorShape = list()
#       for i in inputs["tensorShape"]["dim"]:
#         tensorShape.append(i["size"])
#       input_shapes.append(tensorShape)
#     else:
#       for key in inputs:
#         input_names.append(inputs[key]['name'])
#         input_dtypes.append(inputs[key]['dtype'][key])
#         tensorShape = list()
#         for i in inputs[key]["tensorShape"]["dim"]:
#           tensorShape.append(i["size"])
#         input_shapes.append(tensorShape)

#     for name, shape, dtype in zip(input_names, input_shapes, input_dtypes):
#       shape[0] = None # batching requires the first dimension to be None
#       initial_types.append((name, __NP_DTYPE_TO_SKL_DTYPE[str(dtype)](shape)))
#     onx_model = skl2onnx.to_onnx(self._model, initial_types=initial_types, target_opset=opset_version)

#     with open(export_path, "wb") as f:
#       f.write(onx_model.SerializeToString())

#     return ModelSpec(
#       export_type="onnx",
#       export_path=export_path,
#       load_class=self.__class__.__name__,
#       load_method="from_to_onnx",
#       load_kwargs={
#         "model": f"./{model_file_name}",
#         "map_location": "cpu",
#       },
#       io_dict=iod,
#     )

#   def export_to_pkl(self, input_object, export_model_path, model_file_name = "model.pkl"):
#     export_path = join(export_model_path, model_file_name)
#     with open(export_path, "wb") as f:
#       dill.dump(self._model, f)

#     method = input_object.get("method", None)
#     method = getattr(self._model, "predict") if method == None else getattr(self._model, method)
#     iod = get_io_dict(input_object, method, self.forward)

#     return ModelSpec(
#       export_type = "pkl",
#       load_kwargs = {
#         "model": f"./{model_file_name}",
#         "map_location": "cpu",
#       },
#       io_dict = iod,
#     )

#   def export(self, format, input_object, export_model_path, **kwargs) -> ModelSpec:
#     if format == "onnx":
#       model_spec = self.export_to_onnx(input_object, export_model_path=export_model_path,**kwargs)
#     elif format == "pkl":
#       model_spec = self.export_to_pkl(input_object, export_model_path=export_model_path, **kwargs)
#     else:
#       raise InvalidProtocolError(f"Unsupported export format for torch Module: {format}")

#     import sklearn
#     model_spec.src_framework = "sklearn"
#     model_spec.src_framework_version = sklearn.__version__

#     return model_spec

#   @staticmethod
#   def deserialise(model_meta: ModelSpec) -> Tuple[Any, Any]:
#     logger.info(f"Deserialising SkLearn model")
#     kwargs = model_meta.load_kwargs
#     if model_meta.export_type == "pkl":
#       with open(kwargs["model"], "rb") as f:
#         model = dill.load(f)

#     else:
#       raise InvalidProtocolError(f"Unknown format: {model_meta.export_type}")

#     return model, logic


# ################################################################################
# # Tensorflow
# # ==========
# # Tensorflow is a high-level machine learning library built by Google. It is
# # part of a much larger ecosystem called tensorflow extended.
# ################################################################################

# class TensorflowModel(FrameworkAgnosticProtocol):
#   @U.isthere("tensorflow", soft = False)
#   def __init__(self, m0):
#     import tensorflow as tf
#     logger.debug(f"Trying to load Tensorflow model")
#     if not isinstance(m0, (tf.keras.Model, tf.Module)):
#       raise InvalidProtocolError(f"First input must be a tensorflow module or Keras model, got: {type(m0)}")

#     self._model = m0

#   def forward(self, input_object: Any) -> ModelOutput:
#     model_inputs = self._logic(input_object)
#     if isinstance(model_inputs, dict):
#       out = self._model(list(model_inputs.values()))
#     else:
#       out = self._model(model_inputs)
#     return ModelOutput(inputs = input_object, outputs = out)

#   def _serialise_logic(self, fpath):
#     logger.debug(f"Saving logic to {fpath}")
#     with open(fpath, "wb") as f:
#       dill.dump(self._logic, f)

#   @U.isthere("tf2onnx", soft = False)
#   def export_to_onnx(
#     self,
#     input_object,
#     export_model_path,
#     logic_file_name="logic.dill",
#     model_file_name="model.onnx",
#     opset_version=None,
#   ) -> ModelSpec:
#     import tf2onnx
#     import tensorflow as tf

#     input_object = self._logic(input_object)

#     self._serialise_logic(join(export_model_path, logic_file_name))
#     export_path = join(export_model_path, model_file_name)
#     iod = get_io_dict(input_object, self._model.call, self.forward)

#     if "keras" in str(self._model.__class__):
#       model_proto, _ = tf2onnx.convert.from_keras(self._model, opset=opset_version)
#       with open(export_path, "wb") as f:
#         f.write(model_proto.SerializeToString())

#       return ModelSpec(
#         src_framework="tf",
#         src_framework_version=tf.__version__,
#         export_type="onnx",
#         export_path=export_path,
#         load_class=self.__class__.__name__,
#         load_method="from_convert_from_keras",
#         load_kwargs={
#           "model": f"./{model_file_name}",
#           "map_location": "cpu",
#           "logic_path": f"./{logic_file_name}",
#         },
#         io_dict=iod,
#       )
#     else:
#       raise InvalidProtocolError(f"Unsupported ONNX export for tensorflow model: {type(self._model)}")

#   def export_to_savemodel(
#     self,
#     input_object,
#     export_model_path,
#     logic_file_name = "logic.dill",
#     model_dir_name = "model",
#     include_optimizer = True,
#   ) -> ModelSpec:
#     import tensorflow as tf

#     input_object = self._logic(input_object)
#     self._serialise_logic(join(export_model_path, logic_file_name))

#     export_path = join(export_model_path, model_dir_name)
#     iod = get_io_dict(input_object, self._model.call, self.forward)

#     if "keras" in str(self._model.__class__):
#       tf.keras.models.save_model(
#         self._model,
#         export_path,
#         signatures = None,
#         options = None,
#         include_optimizer = include_optimizer,
#         save_format = "tf"
#       )

#       return ModelSpec(
#         src_framework = "tf",
#         src_framework_version = tf.__version__,
#         export_type = "SaveModel",
#         export_path = export_path,
#         load_class = self.__class__.__name__,
#         load_kwargs = {
#           "model": f"./{model_dir_name}",
#           "map_location": "cpu",
#           "logic_path": f"./{logic_file_name}"
#         },
#         io_dict = iod,
#       )
#     else:
#       raise InvalidProtocolError(f"Unsupported SaveModel export for tensorflow model: {type(self._model)}")

#   def export_to_h5(
#     self,
#     input_object,
#     export_model_path,
#     logic_file_name = "logic.dill",
#     model_file_name = "model.h5py",
#     include_optimizer = True,
#   ) -> ModelSpec:

#     import tensorflow as tf

#     self._serialise_logic(join(export_model_path, logic_file_name))
#     iod = get_io_dict(input_object, self._model.call, self.forward)

#     export_path = join(export_model_path, model_file_name)

#     if "keras" in str(self._model.__class__):
#       tf.keras.models.save_model(
#         self._model,
#         export_path,
#         signatures=None,
#         options=None,
#         include_optimizer=include_optimizer,
#         save_format = "h5"
#       )

#       return ModelSpec(
#         src_framework = "tf",
#         src_framework_version = tf.__version__,
#         export_type = "h5",
#         export_path = export_path,
#         load_class = self.__class__.__name__,
#         load_method = "from_savemodel",
#         load_kwargs = {
#           "model": f"./{model_file_name}",
#           "map_location": "cpu",
#           "logic_path": f"./{logic_file_name}"
#         },
#         io_dict = iod,
#       )
#     else:
#       raise InvalidProtocolError(f"Unsupported h5 export for tensorflow model: {type(self._model)}")

#   def export(self, format: str, input_object: Any, export_model_path: str, **kwargs) -> ModelSpec:
#     if format == "onnx":
#       return self.export_to_onnx(
#         input_object, export_model_path = export_model_path, **kwargs
#       )
#     elif format == "SaveModel":
#       return self.export_to_savemodel(
#         input_object, export_model_path=export_model_path, **kwargs
#       )
#     elif format == "h5":
#       return self.export_to_h5(
#         input_object, export_model_path=export_model_path, **kwargs
#       )
#     else:
#       raise  InvalidProtocolError(f"Unknown format: {format}")

#   @staticmethod
#   def deserialise(model_meta: ModelSpec) -> Tuple[Any, Any]:

#     logger.info(f"Deserialising Tensorflow model from {model_meta.export_path}")
#     kwargs = model_meta.load_kwargs

#     if model_meta.export_type in ["SaveModel", "h5"]:
#       lp = kwargs.pop("logic_path")
#       logger.info(f"Loading logic from {lp}")
#       with open(lp, "rb") as f:
#         logic = dill.load(f)

#       import tensorflow as tf
#       model = tf.keras.models.load_model(model_meta.load_kwargs["model"])
#       return model, logic
#     else:
#       raise InvalidProtocolError(f"Unknown format: {model_meta.export_type}")

# ################################################################################
# # Jax
# # ===
# # Jax is built by Google for high performance ML research. Read more here:
# # https://jax.readthedocs.io/en/latest/notebooks/quickstart.html
# ################################################################################

# """ We are not implementing Jax as of now"""


# ################################################################################
# # Flax
# # ===
# # Flax is built by Google, it is a neural network library and ecosystem for JAX
# # designed for flexibility. Read more here: https://github.com/google/flax
# ################################################################################

# class FlaxModel(FrameworkAgnosticProtocol):
#   @U.isthere("flax", soft = False)
#   def __init__(self, m0):
#     import flax
#     from flax import linen as nn

#     logger.debug("Trying to load Flax model")

#     if not isinstance(m0, nn.Module):
#       raise InvalidProtocolError(f"First input must be a Flax module, got: {type(m0)}")

#     if not isinstance(m1, flax.core.frozen_dict.FrozenDict):
#       # m1 should contain the model parameters
#       raise ValueError(f"Second input for flax model must be a\
#        FrozenDict containing model parameters, got: {type(m1)}")

#     self._model = m0
#     self._params = m1

#   def forward(self, input_object: Any) -> ModelOutput:
#     model_inputs = input_object
#     if type(model_inputs) is dict:
#       out = self._model.apply(self._params, **model_inputs)
#     else:
#       out = self._model.apply(self._params, model_inputs)

#     return ModelOutput(inputs=input_object, outputs=out)

#   def _serialise_params(self, fpath):
#     logger.info(f"Saving parameters to {fpath}")
#     raise NotImplementedError

#   def export(self, format: str, input_object: Any, export_model_path: str, **kwargs) -> ModelSpec:
#    raise NotImplementedError

#   @staticmethod
#   def deserialise(model_meta: ModelSpec) -> Tuple[Any, Any]:
#     raise NotImplementedError
