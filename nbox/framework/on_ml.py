"""
ML Framework
============

I started with the initial assumption that abstracting away the details of
processing the data was doable but unfortunately I was not able to get it to
work reliably. I ended up moving back to configured format because that is
easy to scale and maintain (as @rohanpooniwala keeps saying). All the methods
are descibed in this single file that then has the methods for:

1. ``NBX-Deploy``
2. ``torch``
3. ``sklearn``
4. ``onnruntime``

This has a protocol like behaviour where any node is supposed to execute
certain things in certain fashion. For more information on this read the

Read the code for best understanding.
"""

# weird file name, yessir. Why? because
# from nbox.framework.on_ml import ModelOuput
# read as "from nbox's framework on ML import ModelOutput"

import json
import inspect
import joblib
import requests
from time import time
from typing import Any, Tuple
import dill

from .on_functions import DBase
from ..utils import isthere, join, logger
from ..auth import secret


################################################################################
# Utils
# =====
# The base class FrameworkAgnosticProtocol defines the two things user must define
# in order to use the framework agnostic model:
#   1. `forward` method: the forward pass of the model
#   2. `export`: that takes in a string for export format and arguments for it
# Other functions are self-explanatory
################################################################################

class InvalidProtocolError(Exception):
  pass

class ModelOutput(DBase):
  __slots__ = [
    "inputs", # :Any
    "outputs", # :Any
  ]

class ModelSpec(DBase):
  __slots__ = [
    # where from
    "src_framework", # :str: name of the source framework
    "src_framework_version", # :str
    'export_path', # str: there is no reason for the Pod to know anything about the user

    # where to
    "export_type", # :str
    "exported_time", # :str: UTC time when the model was exported

    # how to
    "load_method", # :str: The classmethod to call to load this model
    "load_kwargs", # :dict: kwargs to pass to the load method
    "io_dict", # :dict: obtained from above function

    # 'what to' is the serving script!
  ]

class FrameworkAgnosticProtocol(object):
  """https://nimblebox.notion.site/nbox-FrameworkAgnosticProtocol-6b39249316b1497b8ad9ff8f02b227f0"""
  # def __init__(self, i0: Any, i1: Any) -> None

  def forward(self, input_object: Any) -> ModelOutput:
    raise NotImplementedError()

  def export(self, format: str, input_object: Any, export_model_path: str, **kwargs) -> ModelSpec:
    raise NotImplementedError()

  @staticmethod
  def deserialise(self, model_meta: ModelSpec) -> Tuple[Any, Any]:
    raise NotImplementedError()


def io_dict(input_object, output_object):
  """Generic method to convert the inputs to get ``nbox_meta['metadata']`` dictionary"""
  # get the meta object

  def __get_struct(object):

    def process_dict(x, curr_idx) -> dict:
      #Process Dictionaries
      results = {}
      for key in x.keys():
        parsed, curr_idx = parse(x[key], name=key, curr_idx=curr_idx)
        results[key] = parsed
      return results, curr_idx

    def process_container(x, curr_idx) -> dict:
      #Handles lists, sets and tuples
      results = []
      for element in x:
        parsed, curr_idx = parse(element, None, curr_idx)
        results.append(parsed)

      return results, curr_idx

    def parse(x, name=None, curr_idx=0) -> dict:
      # Parses objects to generate iodict
      dtype = None
      if name is None:
        name = f"tensor{curr_idx}"
        curr_idx += 1

      if hasattr(x, "dtype"):
        dtype = str(x.dtype)

      if isinstance(x, dict):
        return process_dict(x, curr_idx)

      elif isinstance(x, (list, set, tuple)):
        return process_container(x, curr_idx)

      elif hasattr(x, "shape"):
        dim_names=[""]*len(x.shape)
        if hasattr(x, "names"):
          dim_names=x.names
        return {"name": name, "dtype": dtype, "tensorShape": {"dim":[{'name':dim_names[y], "size":x.shape[y]} for y in range(len(x.shape))],"unknownRank":False}}, curr_idx
      else:
        return {"name": name, "dtype": dtype, "shape": None}, curr_idx

    return parse(object)[0]

  meta = {
    "inputs": __get_struct(input_object),
    "outputs": __get_struct(output_object)
  }
  return meta

def get_io_dict(input_object, call_fn, forward_fn):
  """Generates and returns an io_dict by performing forward pass through the model

  Args:
      input_object (Any): Input to a model
      call_fn (Callable): function that Model.model employs to do a forward pass.
      forward_fn (Callable): forward() function of the Model

  Returns:
      io : io_dict
  """
  out = forward_fn(input_object)
  args = inspect.getfullargspec(call_fn)
  args.args.remove("self")

  io = io_dict(
    input_object=out.inputs,
    output_object=out.outputs
  )
  io["arg_spec"] = {
    "args": args.args,
    "varargs": args.varargs,
    "varkw": args.varkw,
    "defaults": args.defaults,
    "kwonlyargs": args.kwonlyargs,
    "kwonlydefaults": args.kwonlydefaults,
  }
  return io


################################################################################
# NimbleBox.ai Deployments
# ========================
# NBX-Deploy is a serive that you can use to load the API endpoints where it is
# deployed. This is a special kind of method because it can consume all the
# objects that other models can consume.
################################################################################

# TODO:@yashbonde
class NBXModel(FrameworkAgnosticProtocol):
  @isthere("numpy", soft=False)
  def __init__(self, url, key):
    self.url = url
    self.key = key

    logger.debug(f"Trying to load as url")
    if not isinstance(url, str):
      raise InvalidProtocolError(f"Model must be a string, got: {type(url)}")
    if not (url.startswith("https://") or url.startswith("http://")):
      raise InvalidProtocolError("Model URL must start with http:// or https://")
    if not isinstance(key, str):
      raise InvalidProtocolError("Nbx API key must be a string")
    if not key.startswith("nbxdeploy_"):
      raise InvalidProtocolError("Not a valid NBX Api key, please check again.")

    # fetch the metadata from the cloud
    model_url = url.rstrip("/")
    logger.debug("Getting model metadata")
    URL = secret.get("nbx_url")
    r = requests.get(f"{URL}/api/model/get_model_meta", params=f"url={model_url}&key={key}")
    try:
      r.raise_for_status()
    except:
      raise ValueError(f"Could not fetch metadata, please check status: {r.status_code}")

    # start getting the metadata, note that we have completely dropped using OVMS meta and instead use nbox_meta
    content = json.loads(r.content.decode())["meta"]
    nbox_meta = content["nbox_meta"]

    all_inputs = nbox_meta["metadata"]["inputs"]
    templates = {}
    for node, meta in all_inputs.items():
      templates[node] = [int(x["size"]) for x in meta["tensorShape"]["dim"]]
    logger.debug("Cloud infer metadata obtained")

    category = nbox_meta["spec"]["category"]

    # if category is "text" or if it is dict then any key is "text"
    tokenizer = None
    max_len = None
    if category == "text" or (isinstance(category, dict) and any([x == "text" for x in category.values()])):
      import transformers

      model_key = nbox_meta["spec"]["model_key"].split("::")[0].split("transformers/")[-1]
      tokenizer = transformers.AutoTokenizer.from_pretrained(model_key)
      max_len = templates["input_ids"][-1]

    image_parser = ImageParser(cloud_infer=True, post_proc_fn=lambda x: x.tolist(), templates=templates)
    text_parser = TextParser(tokenizer=tokenizer, max_len=max_len, post_proc_fn=lambda x: x.tolist())

  def forward(self, model_input):
    import numpy as np

    logger.debug(f"Hitting API: {self.model_or_model_url}")
    st = time()
    # OVMS has :predict endpoint and nbox has /predict
    _p = "/" if "export_type" in self.nbox_meta["spec"] else ":"
    json = {"inputs": model_input}
    if "export_type" in self.nbox_meta["spec"]:
      json["method"] = method
    r = requests.post(self.url + f"/{_p}predict", json=json, headers={"NBX-KEY": self.key})
    et = time() - st
    out = None

    try:
      r.raise_for_status()
      out = r.json()

      # first try outputs is a key and we can just get the structure from the list
      if isinstance(out["outputs"], dict):
        out = {k: np.array(v) for k, v in r.json()["outputs"].items()}
      elif isinstance(out["outputs"], list):
        out = np.array(out["outputs"])
      else:
        raise ValueError(f"Outputs must be a dict or list, got {type(out['outputs'])}")
      logger.debug(f"Took {et:.3f} seconds!")
    except Exception as e:
      logger.debug(f"Failed: {str(e)} | {r.content.decode()}")

  def export(*_, **__):
    raise InvalidProtocolError("NBX-Deploy does not support exporting")

  def deserialise(*_, **__):
    raise InvalidProtocolError("NBX-Deploy cannot be loaded by deserialisation, use __init__")


################################################################################
# Torch
# =====
# Torch is a deep learning framework from Facebook. It is aimed towards a simple
# interface and is a huge inspiration to me. User needs to define the
# config/init and forward pass. This is the minimum input you need to take from
# a user to get all the output
################################################################################

class TorchModel(FrameworkAgnosticProtocol):
  @isthere("torch", soft=False)
  def __init__(self, m1, m2):
    import torch

    if not isinstance(m1, torch.nn.Module):
      raise InvalidProtocolError(f"First input must be a torch model, got: {type(m1)}")
    if m2 != None and not callable(m2):
      # this is the processing logic for the incoming data this has to be a some kind of callable,
      # because user can define whatever they want and it removes any onus on NBX to ensure that
      # processing works, halting problem is no joke!
      raise ValueError(f"Second input for torch model must be a callable, got: {type(m2)}")

    self._model = m1
    self._logic = m2 if m2 else lambda x: x # Identity function

  def train(self):
    self._model.train()

  def eval(self):
    self._model.eval()

  def forward(self, input_object) -> ModelOutput:
    model_inputs = self._logic(input_object) # TODO: @yashbonde enforce logic to return dict
    if isinstance(model_inputs, dict):
      out = self._model(**model_inputs)
    else:
      out = self._model(model_inputs)
    return ModelOutput(inputs = input_object, outputs = out)

  def _serialise_logic(self, fpath):
    logger.debug(f"Saving logic to {fpath}")
    with open(fpath, "wb") as f:
      dill.dump(self._logic, f)

  # TODO:@yashbonde
  def export_to_onnx(
    self,
    input_object,
    export_model_path,
    export_params=True,
    verbose=False,
    logic_file_name = "logic.dill",
    model_file_name="model.onnx",
    opset_version=12,
    do_constant_folding=True,
    use_external_data_format=False,
    **kwargs
  ) -> ModelSpec:
    iod = get_io_dict(input_object, self._model.forward, self.forward)
    self._serialise_logic(join(export_model_path, logic_file_name))

    export_path = join(export_model_path, model_file_name)

    import torch
    torch.onnx.export(
      self._model,
      input_object,
      f=export_path,
      verbose=verbose,
      use_external_data_format=use_external_data_format, # RuntimeError: Exporting model exceed maximum protobuf size of 2GB
      export_params=export_params, # store the trained parameter weights inside the model file
      opset_version=opset_version, # the ONNX version to export the model to
      do_constant_folding=do_constant_folding, # whether to execute constant folding for optimization
    )

    return ModelSpec(
        src_framework = "torch",
        src_framework_version = torch.__version__,
        export_type = "onnx",
        export_path = export_model_path,
        load_class = self.__class__.__name__,
        load_kwargs = {
          "model": f"./{model_file_name}",
          "map_location": "cpu",
          "logic_path": f"./{logic_file_name}"
        },
        io_dict = iod,
        )

  def export_to_torchscript(
    self,
    input_object,
    export_model_path,
    check_tolerance = 1e-4,
    optimize = None,
    check_trace = True,
    check_inputs = None,
    strict = True,

    logic_file_name = "logic.dill",
    model_file_name = "model.bin",
  ) -> ModelSpec:
    iod = get_io_dict(input_object, self._model.forward, self.forward)

    self._serialise_logic(join(export_model_path, logic_file_name))

    import torch

    args = self._logic(input_object)

    if isinstance(args, dict):
      args = tuple(args.values())

    traced_model = torch.jit.trace(
      func = self._model,
      example_inputs = args,
      check_tolerance = check_tolerance,
      optimize = optimize,
      check_trace = check_trace,
      check_inputs = check_inputs,
      strict = strict,
    )

    export_path = join(export_model_path, model_file_name)
    logger.debug(f"Saving model to {export_path}")

    torch.jit.save(traced_model, export_path)

    # return as
    return ModelSpec(
        src_framework = "torch",
        src_framework_version = torch.__version__,
        export_type = "torchscript",
        export_path = export_path,
        load_class = self.__class__.__name__,
        load_kwargs = {
          "model": f"./{model_file_name}",
          "map_location": "cpu",
          "logic_path": f"./{logic_file_name}"
        },
        io_dict = iod,
      )

  def export(self, format, input_object, export_model_path, **kwargs) -> ModelSpec:
    logger.debug(f"Exporting torch model to {format}")

    if format == "onnx":
      return self.export_to_onnx(
        input_object = input_object, export_model_path = export_model_path, **kwargs
      )
    elif format == "torchscript":
      return self.export_to_torchscript(
        input_object = input_object, export_model_path = export_model_path, **kwargs
      )
    else:
      raise InvalidProtocolError(f"Unknown format: {format}")

  @staticmethod
  def deserialise(model_meta: ModelSpec) -> Tuple[Any, Any]:
    logger.debug(f"Deserialising torch model from {model_meta.export_path}")

    kwargs = model_meta.load_kwargs

    if model_meta.export_type == "torchscript":
      lp = kwargs.pop("logic_path")
      logger.debug(f"Loading logic from {lp}")
      with open(lp, "rb") as f:
        logic = dill.load(f)

      import torch

      model = torch.jit.load(model_meta.load_kwargs["model"], map_location=kwargs["map_location"])
      return model, logic

    else:
      raise InvalidProtocolError(f"Unknown format: {model_meta.export_type}")


################################################################################
# ONNX Runtime
# ============
# OnnxRuntime is a C++ library that allows you to run ONNX models in your
# application. It is a wrapper around the ONNX API. It *does not*
# work on Apple M1 machines and so suck it up! But this is useful for server
# side processing.
################################################################################

# TODO:@yashbonde
class ONNXRtModel(FrameworkAgnosticProtocol):
  @isthere("onnxruntime", soft = False)
  def __init__(self, m0, m1):
    import onnx
    import onnxruntime as ort
    if not isinstance(m0, onnx.onnx_ml_pb2.ModelProto):
      raise InvalidProtocolError(f"First input must be a ONNX model, got: {type(m0)}")

    if m1!= None and not callable(m1):
      # this is the processing logic for the incoming data this has to be a some kind of callable,
      # because user can define whatever they want and it removes any onus on NBX to ensure that
      # processing works, halting problem is no joke!
      raise ValueError(f"Second input for torch model must be a callable, got: {type(m1)}")

    self._model = m0
    self._logic = m1 if m1 else lambda x: x
    self._sess = ort.InferenceSession(self._model.SerializeToString())


  def forward(self, input_object) -> ModelOutput:
    import numpy as np
    model_inputs = self._logic(input_object)
    input_names = [i.name for i in self._sess.get_inputs()]
    output_names = [o.name for o in self._sess.get_outputs()]
    if isinstance(model_inputs,dict):
      tensor_dict = model_inputs
      if isinstance(list(model_inputs.values())[0],SklearnInput):
        tensor_dict = {list(model_inputs.keys())[0]:list(model_inputs.values())[0].inputs}
      model_inputs = {name: tensor for name, tensor in zip(input_names, tuple(tensor_dict.values()))}
    else:
      raise ValueError("Your Logic needs to return dictionary")

    # conversion to proper arrays -> if can be numpy-ed and float32
    for key, value in model_inputs.items():
      if hasattr(value, "numpy"):
        value = value.numpy()
      if value.dtype in (np.float64, np.float32, np.float16):
        value = value.astype(np.float32)
      elif value.dtype in (np.int64, np.int32, np.int16, np.int8, np.int64):
        value = value.astype(np.int32)
      model_inputs[key] = value

    res = self._sess.run(output_names=output_names, input_feed=model_inputs)
    return ModelOutput(
      inputs = input_object,
      outputs = res,
    )

  def _serialise_logic(self, fpath):
    logger.debug(f"Saving logic to {fpath}")
    with open(fpath, "wb") as f:
      dill.dump(self._logic, f)

  def export(*_, **__):
    raise InvalidProtocolError("ONNXRtModel cannot be exported")

  @staticmethod
  def deserialise(model_meta: ModelSpec) -> Tuple[Any, Any]:
    logger.debug(f"Deserialising ONNX model from {model_meta.export_path}")

    kwargs = model_meta.load_kwargs

    if model_meta.export_type == "onnx":
      if "logic_path" in kwargs:
        lp = kwargs.pop("logic_path")
      else:
        lp = kwargs.pop("method_path")
      logger.debug(f"Loading logic from {lp}")
      with open(lp, "rb") as f:
        logic = dill.load(f)

      import onnx
      model = onnx.load(model_meta.load_kwargs["model"])
      return model, logic

    else:
      raise InvalidProtocolError(f"Unknown format: {model_meta.export_type}")


################################################################################
# Sklearn
# =======
# Scikit-learn is a Python library that allows you to use classical machine
# learning algorithms in your application. Its versatility is huge and that
# means that a dedicate Input DataModel is required.
################################################################################

# TODO:@yashbonde
class SklearnInput(DBase):
  __slots__ = [
    "inputs", # :Any
    "method", # :str
    "kwargs", # :Dict
  ]

class SklearnModel(FrameworkAgnosticProtocol):
  @isthere("sklearn", "numpy", soft = False)
  def __init__(self, m1, m2):
    if not "sklearn" in str(type(m1)):
      raise InvalidProtocolError
    if m2 != None and not callable(m2):
      # this is the processing logic for the incoming data this has to be a some kind of callable,
      # because user can define whatever they want and it removes any onus on NBX to ensure that
      # processing works, halting problem is no joke!
      InvalidProtocolError(f"Second input must be a callable, got: {type(m2)}")

    self._model = m1
    self._logic = m2 # this is not used anywhere

  def forward(self, input_object: SklearnInput) -> ModelOutput:
    model_input = input_object.get("inputs", None)
    method = input_object.get("method", None)
    extra_args = input_object.get("kwargs", {})

    if "sklearn.neighbors.NearestNeighbors" in str(type(self._model)):
      method = getattr(self._model, "kneighbors") if method == None else getattr(self._model, method)
      out = method(model_input, **extra_args)
    elif "sklearn.cluster" in str(type(self._model)):
      if any(
        x in str(type(self._model)) for x in ["AgglomerativeClustering", "DBSCAN", "OPTICS", "SpectralClustering"]
      ):
        method = getattr(self._model, "fit_predict")
        out = method(model_input)
    else:
      try:
        method = getattr(self._model, "predict") if method == None else getattr(self._model, method)
        out = method(model_input)
      except Exception as e:
        logger.debug(f"[ERROR] Model Prediction Function is not yet registered {e}")

    return ModelOutput(
      inputs = model_input,
      outputs = out,
    )

  def _serialise_method(self, fpath):
    logger.info(f"Saving logic to {fpath}")
    with open(fpath, "wb") as f:
      dill.dump(self._logic, f)

  @isthere("skl2onnx", soft = False)
  def export_to_onnx(self, input_object, export_model_path, method_file_name="logic.dill",
                           model_file_name="model.onnx",
                           opset_version=None,
                    ) -> ModelSpec:
    import sklearn
    from skl2onnx import to_onnx
    import skl2onnx.common.data_types as dt

    __NP_DTYPE_TO_SKL_DTYPE = {
      "bool_": dt.BooleanTensorType,
      "double": dt.DoubleTensorType,
      "int16": dt.Int16TensorType,
      "complex128": dt.Complex128TensorType,
      "int32": dt.Int32TensorType,
      "uint16": dt.UInt16TensorType,
      "complex64": dt.Complex64TensorType,
      "float16": dt.Float16TensorType,
      "int64": dt.Int64TensorType,
      "str_": dt.StringTensorType,
      "uint32": dt.UInt32TensorType,
      "float64": dt.FloatTensorType,
      "uint64": dt.UInt64TensorType,
      "int8": dt.Int8TensorType,
      "uint8": dt.UInt8TensorType,
    }

    self._serialise_method(join(export_model_path, method_file_name))

    export_path = join(export_model_path, model_file_name)
    method = input_object.get("method", None)
    method = getattr(self._model, "predict") if method == None else getattr(self._model, method)
    iod = get_io_dict(input_object, method, self.forward)

    # the args need to be converted to proper formatted data types
    initial_types = []
    input_names, input_shapes, input_dtypes = [],[],[]

    # Case for input object with single tensor
    inputs=iod["inputs"]
    if type(inputs['name'])==str:
      input_names.append(inputs["name"])
      input_dtypes.append(inputs["dtype"])
      tensorShape = list()
      for i in inputs["tensorShape"]["dim"]:
        tensorShape.append(i["size"])
      input_shapes.append(tensorShape)

    else:
      for key in inputs:
          input_names.append(inputs[key]['name'])
          input_dtypes.append(inputs[key]['dtype'][key])
          tensorShape = list()
          for i in inputs[key]["tensorShape"]["dim"]:
            tensorShape.append(i["size"])
          input_shapes.append(tensorShape)

    for name, shape, dtype in zip(input_names, input_shapes, input_dtypes):
      shape[0] = None # batching requires the first dimension to be None
      initial_types.append((name, __NP_DTYPE_TO_SKL_DTYPE[str(dtype)](shape)))
    onx = to_onnx(self._model, initial_types=initial_types, target_opset=opset_version)

    with open(export_path, "wb") as f:
      f.write(onx.SerializeToString())


    return ModelSpec(
    src_framework="sklearn",
    src_framework_version=sklearn.__version__,
    export_type="onnx",
    export_path=export_path,
    load_class=self.__class__.__name__,
    load_method="from_to_onnx",
    load_kwargs={
        "model": f"./{model_file_name}",
        "map_location": "cpu",
        "method_path": f"./{method_file_name}",
    },
    io_dict=iod,
    )

  def export_to_pkl(self, input_object, export_model_path, method_file_name = "method.dill", model_file_name = "model.pkl"):

    import joblib
    import sklearn
    # sklearn models are pure python methods (though underlying contains bindings to C++)
    # and so we can use joblib for this
    # we use the joblib instead of pickle
    self._serialise_method(join(export_model_path, method_file_name))
    export_path = join(export_model_path, model_file_name)
    with open(export_path, "wb") as f:
      joblib.dump(self._model, f)

    method = input_object.get("method", None)
    method = getattr(self._model, "predict") if method == None else getattr(self._model, method)
    iod = get_io_dict(input_object, method, self.forward)

    return ModelSpec(
      src_framework = "SkLearn",
      src_framework_version = sklearn.__version__,
      export_type = "pkl",
      export_path = export_path,
      load_class = self.__class__.__name__,
      load_kwargs = {
        "model": f"./{model_file_name}",
        "map_location": "cpu",
       "method_path": f"./{method_file_name}"
      },
      io_dict = iod,
    )

  def export(self, format, input_object, export_model_path, **kwargs) -> ModelSpec:
    if format == "onnx":
      return self.export_to_onnx(input_object, export_model_path=export_model_path,**kwargs)
    elif format == "pkl":
      return self.export_to_pkl(input_object, export_model_path=export_model_path, **kwargs)
    else:
      raise InvalidProtocolError(f"Unsupported export format for torch Module: {format}")

  @staticmethod
  def deserialise(model_meta: ModelSpec) -> Tuple[Any, Any]:
    logger.info(f"Deserialising SkLearn model from {model_meta.export_path}")
    kwargs = model_meta.load_kwargs
    if model_meta.export_type == "pkl":
      lp = kwargs.pop("method_path")
      logger.info(f"Loading method from {lp}")
      with open(lp, "rb") as f:
        method = dill.load(f)
      model = joblib.load(model_meta.load_kwargs["model"])

    else:
      raise InvalidProtocolError(f"Unknown format: {model_meta.export_type}")

    return model, method

################################################################################
# Tensorflow
# ==========
# Tensorflow is a high-level machine learning library built by Google. It is
# part of a much larger ecosystem called tensorflow extended.
################################################################################

class TensorflowModel(FrameworkAgnosticProtocol):
  @isthere("tensorflow", soft = False)
  def __init__(self, m0, m1):
    import tensorflow as tf
    if not isinstance(m0, (tf.keras.Model, tf.Module)):
      raise InvalidProtocolError(f"First input must be a tensorflow module or Keras model, got: {type(m0)}")

    if m1!= None and not callable(m1):
      # this is the processing logic for the incoming data this has to be a some kind of callable,
      # because user can define whatever they want and it removes any onus on NBX to ensure that
      # processing works, halting problem is no joke!
      raise ValueError(f"Second input for torch model must be a callable, got: {type(m1)}")

    self._model = m0
    self._logic = m1 if m1 else lambda x: x

  def forward(self, input_object: Any) -> ModelOutput:
    model_inputs = self._logic(input_object)
    if isinstance(model_inputs, dict):
      out = self._model(**model_inputs)
    else:
      out = self._model(model_inputs)
    return ModelOutput(inputs = input_object, outputs = out)


  def _serialise_logic(self, fpath):
    logger.info(f"Saving logic to {fpath}")

  def _serialise_logic(self, fpath):
    logger.debug(f"Saving logic to {fpath}")
    with open(fpath, "wb") as f:
      dill.dump(self._logic, f)

  @isthere("tf2onnx", soft = False)
  def export_to_onnx(
    self,
    input_object,
    export_model_path,
    logic_file_name="logic.dill",
    model_file_name="model.onnx",
    opset_version=None,
) -> ModelSpec:
    import tf2onnx
    import tensorflow as tf

    input_object = self._logic(input_object)

    self._serialise_logic(join(export_model_path, logic_file_name))
    export_path = join(export_model_path, model_file_name)
    iod = get_io_dict(input_object, self._model.call, self.forward)

    if "keras" in str(self._model.__class__):
        model_proto, _ = tf2onnx.convert.from_keras(
            self._model, opset=opset_version
        )

        with open(export_path, "wb") as f:
              f.write(model_proto.SerializeToString())


        return ModelSpec(
            src_framework="tf",
            src_framework_version=tf.__version__,
            export_type="onnx",
            export_path=export_path,
            load_class=self.__class__.__name__,
            load_method="from_convert_from_keras",
            load_kwargs={
                "model": f"./{model_file_name}",
                "map_location": "cpu",
                "logic_path": f"./{logic_file_name}",
            },
            io_dict=iod,
        )


  def export_to_savemodel(
    self,
    input_object,
    export_model_path,
    logic_file_name = "logic.dill",
    model_dir_name = "model",
    include_optimizer = True,
  ) -> ModelSpec:
    import tensorflow as tf

    input_object = self._logic(input_object)
    self._serialise_logic(join(export_model_path, logic_file_name))

    export_path = join(export_model_path, model_dir_name)

    tf.keras.models.save_model(
    self._model, export_path, signatures=None,
     options=None, include_optimizer=include_optimizer, save_format = "tf"
    )

    iod = get_io_dict(input_object, self._model.call, self.forward)

    return ModelSpec(
      src_framework = "tf",
      src_framework_version = tf.__version__,
      export_type = "SaveModel",
      export_path = export_path,
      load_class = self.__class__.__name__,
      load_kwargs = {
        "model": f"./{model_dir_name}",
        "map_location": "cpu",
        "logic_path": f"./{logic_file_name}"
      },
      io_dict = iod,
    )

  def export_to_h5(
    self,
    input_object,
    export_model_path,
    logic_file_name = "logic.dill",
    model_file_name = "model.h5py",
    include_optimizer = True,
  ) -> ModelSpec:

    import tensorflow as tf

    self._serialise_logic(join(export_model_path, logic_file_name))

    export_path = join(export_model_path, model_file_name)

    tf.keras.models.save_model(
    self._model, export_path, signatures=None,
     options=None, include_optimizer=include_optimizer, save_format = "h5"
    )

    iod = get_io_dict(input_object, self._model.call, self.forward)

    return ModelSpec(
      src_framework = "tf",
      src_framework_version = tf.__version__,
      export_type = "h5",
      export_path = export_path,
      load_class = self.__class__.__name__,
      load_method = "from_savemodel",
      load_kwargs = {
        "model": f"./{model_file_name}",
        "map_location": "cpu",
        "logic_path": f"./{logic_file_name}"
      },
      io_dict = iod,
    )


  def export(self, format: str, input_object: Any, export_model_path: str, **kwargs) -> ModelSpec:
    if format == "onnx":
      return self.export_to_onnx(
        input_object, export_model_path = export_model_path, **kwargs
      )
    elif format == "SaveModel":
      return self.export_to_savemodel(
        input_object, export_model_path=export_model_path, **kwargs
      )
    elif format == "h5":
      return self.export_to_h5(
        input_object, export_model_path=export_model_path, **kwargs
      )
    else:
      raise  InvalidProtocolError(f"Unknown format: {format}")

  @staticmethod
  def deserialise(model_meta: ModelSpec) -> Tuple[Any, Any]:

    logger.info(f"Deserialising Tensorflow model from {model_meta.export_path}")
    kwargs = model_meta.load_kwargs

    if model_meta.export_type in ["SaveModel", "h5"]:
      lp = kwargs.pop("logic_path")
      logger.info(f"Loading logic from {lp}")
      with open(lp, "rb") as f:
        logic = dill.load(f)

      import tensorflow as tf
      model = tf.keras.models.load_model(model_meta.load_kwargs["model"])
      return model, logic

    logger.debug(f"Deserialising Tensorflow model from {model_meta.export_path}")
    kwargs = model_meta.load_kwargs

    if model_meta.export_type in ["SaveModel", "h5"]:
      lp = kwargs.pop("logic_path")
      logger.debug(f"Loading logic from {lp}")
      with open(lp, "rb") as f:
        logic = dill.load(f)

      import tensorflow as tf # pylint: disable=import-outside-toplevel
      model = tf.keras.models.load_model(model_meta.load_kwargs["model"])
      return model, logic

    else:
      raise InvalidProtocolError(f"Unknown format: {model_meta.export_type}")

################################################################################
# Jax
# ===
# Jax is built by Google for high performance ML research. Read more here:
# https://jax.readthedocs.io/en/latest/notebooks/quickstart.html
################################################################################

""" We are not implementing Jax as of now"""

# class JaxModel(FrameworkAgnosticProtocol):
#   @isthere("jax", soft = False)
#   def __init__(self, m0, m1):
#     pass

#   def forward(self, input_object: Any) -> ModelOutput:
#     raise NotImplementedError()

#   def export(self, format: str, input_object: Any, export_model_path: str, **kwargs) -> ModelSpec:
#     raise NotImplementedError()

#   @staticmethod
#   def deserialise(self, model_meta: ModelSpec) -> Tuple[Any, Any]:
#     raise NotImplementedError()


################################################################################
# Flax
# ===
# Flax is built by Google, it is a neural network library and ecosystem for JAX designed for flexibility. Read more here:
# https://github.com/google/flax
################################################################################

class FlaxModel(FrameworkAgnosticProtocol):
  @isthere("flax", soft = False)
  def __init__(self, m0, m1):
    import flax
    from flax import linen as nn

    if not isinstance(m0, nn.Module):
      raise InvalidProtocolError(f"First input must be a Flax module, got: {type(m0)}")

    if not isinstance(m1, flax.core.frozen_dict.FrozenDict):
      # m1 should contain the model parameters
      raise ValueError(f"Second input for flax model must be a\
       FrozenDict containing model parameters, got: {type(m1)}")

    self._model = m0
    self._params = m1


  def forward(self, input_object: Any) -> ModelOutput:
    model_inputs = input_object
    if type(model_inputs) is dict:
      out = self._model.apply(self._params, **model_inputs)
    else:
      out = self._model.apply(self._params, model_inputs)

    return ModelOutput(inputs=input_object, outputs=out)

  def _serialise_params(self, fpath):
    logger.info(f"Saving parameters to {fpath}")
    raise NotImplementedError

  def export(self, format: str, input_object: Any, export_model_path: str, **kwargs) -> ModelSpec:
   raise NotImplementedError

  @staticmethod
  def deserialise(model_meta: ModelSpec) -> Tuple[Any, Any]:
    raise NotImplementedError
