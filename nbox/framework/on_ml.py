"""
ML Models
=========

I started with the initial assumption that abstracting away the details of
processing the data was doable but unfortunately I was not able to get it to
work. I ended up moving back to configured format because that is easy to
scale and maintain (as @rohanpooniwala keeps saying). All the methods are
descibed in this single file that then has the methods for:

1. ``torch``
2. ``sklearn``
3. ``onnruntime``

Read the code for best understanding.
"""

# weird file name, yessir. Why? because
# from nbox.framework.on_ml import ModelOuput
# read as "from nbox's framework on ML import ModelOutput"

import json
import requests
from time import time

from logging import getLogger
logger = getLogger()

from .on_functions import DBase
from ..utils import isthere
from ..auth import secret


################################################################################
# Utils
# =====
# The base class FrameworkAgnosticModel defines the two things user must define
# in order to use the framework agnostic model:
#   1. `forward` method: the forward pass of the model
#   2. `export`: that takes in a string for export format and arguments for it
################################################################################

class IllegalFormatError(Exception):
  pass

class ModelOutput(DBase):
  __slots__ = [
    "inputs", # :Any
    "outputs", # :Any:
  ]

class FrameworkAgnosticModel(object):
  def forward(self) -> ModelOutput:
    pass

  def export(self, format: str) -> None:
    pass


################################################################################
# NimbleBox.ai Deployments
# ========================
# NBX-Deploy is a serive that you can use to load the API endpoints where it is
# deployed. This is a special kind of method because it can consume all the
# objects that other models can consume.
################################################################################

# TODO:@yashbonde
class NBXModel(FrameworkAgnosticModel):
  @isthere("numpy", soft=False)
  def __init__(self, url, key):
    self.url = url
    self.key = key

    logger.info(f"Trying to load as url")
    if not isinstance(url, str):
      raise IllegalFormatError(f"Model must be a string, got: {type(url)}")
    if not (url.startswith("https://") or url.startswith("http://")):
      raise IllegalFormatError("Model URL must start with http:// or https://")
    if not isinstance(key, str):
      raise IllegalFormatError("Nbx API key must be a string")
    if not key.startswith("nbxdeploy_"):
      raise IllegalFormatError("Not a valid NBX Api key, please check again.")

    # fetch the metadata from the cloud
    model_url = url.rstrip("/")
    logger.info("Getting model metadata")
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
    logger.info("Cloud infer metadata obtained")

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
    
    logger.info(f"Hitting API: {self.model_or_model_url}")
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
      logger.info(f"Took {et:.3f} seconds!")
    except Exception as e:
      logger.info(f"Failed: {str(e)} | {r.content.decode()}")


################################################################################
# Torch
# =====
# Torch is a deep learning framework from Facebook. It is aimed towards a simple
# interface and is a huge inspiration to me. User needs to define the
# config/init and forward pass. This is the minimum input you need to take from
# a user to get all the output
################################################################################

class TorchModel(FrameworkAgnosticModel):
  def __init__(self, m1, m2):
    import torch

    if not isinstance(m1, torch.nn.Module):
      IllegalFormatError(f"First input must be a torch model, got: {type(m1)}")
    if m2 != None and not callable(m2):
      # this is the processing logic for the incoming data this has to be a some kind of callable,
      # because user can define whatever they want and it removes any onus on NBX to ensure that
      # processing works, halting problem is no joke!
      IllegalFormatError(f"Second input must be a callable, got: {type(m2)}")
    
    self._model = m1
    self._logic = m2

  def forward(self, input_object) -> ModelOutput:
    model_inputs = self._logic(input_object) if self._logic != None else input_object
    out = self._model(**model_inputs)
    return ModelOutput(inputs = input_object, outputs = out)

  def export_to_onnx(
    self,
    model,
    args,
    export_model_path,
    input_names,
    dynamic_axes,
    output_names,
    export_params=True,
    verbose=False,
    opset_version=12,
    do_constant_folding=True,
    use_external_data_format=False,
    **kwargs
  ):
    import torch

    torch.onnx.export(
      model,
      args=args,
      f=export_model_path,
      input_names=input_names,
      verbose=verbose,
      output_names=output_names,
      use_external_data_format=use_external_data_format, # RuntimeError: Exporting model exceed maximum protobuf size of 2GB
      export_params=export_params, # store the trained parameter weights inside the model file
      opset_version=opset_version, # the ONNX version to export the model to
      do_constant_folding=do_constant_folding, # whether to execute constant folding for optimization
      dynamic_axes=dynamic_axes,
    )

  def export_to_torchscript(self, model, args, export_model_path, **kwargs):
    import torch

    traced_model = torch.jit.trace(model, args, check_tolerance=0.0001)
    torch.jit.save(traced_model, export_model_path)

  def export(self, format, *a, **b) -> None:
    if format == "onnx":
      self.export_to_onnx(*a, **b)
    elif format == "torchscript":
      self.export_to_torchscript(*a, **b)
    else:
      raise IllegalFormatError(f"Unsupported export format for torch Module: {format}")


################################################################################
# ONNX Runtime
# ============
# OnnxRuntime is a C++ library that allows you to run ONNX models in your
# application. It is a wrapper around the ONNX API. It *does not*
# work on Apple M1 machines and so suck it up! But this is useful for server
# side processing.
################################################################################

# TODO:@yashbonde
class ONNXRtModel(FrameworkAgnosticModel):
  @isthere("onnxruntime", soft = False)
  def __init__(self, ort_session, nbox_meta):
    import onnxruntime
    
    if not isinstance(ort_session, onnxruntime.InferenceSession):
      raise IllegalFormatError

    logger.info(f"Trying to load from onnx model: {ort_session}")

    # we have to create templates using the nbox_meta
    templates = None
    if nbox_meta is not None:
      all_inputs = nbox_meta["metadata"]["inputs"]
      templates = {}
      for node, meta in all_inputs.items():
        templates[node] = [int(x["size"]) for x in meta["tensorShape"]["dim"]]

    image_parser = ImageParser(post_proc_fn=lambda x: x.astype(np.float32), templates = templates)
    text_parser = TextParser(tokenizer=tokenizer, post_proc_fn=lambda x: x.astype(np.int32))

    self.session = ort_session
    self.input_names = [x.name for x in self.session.get_inputs()]
    self.output_names = [x.name for x in self.session.get_outputs()]

    logger.info(f"Inputs: {self.input_names}")
    logger.info(f"Outputs: {self.output_names}")


  def forward(self, input_object) -> ModelOutput:
    if set(input_object.keys()) != set(self.input_names):
      diff = set(input_object.keys()) - set(self.input_names)
      return f"model_input keys do not match input_name: {diff}"
    out = self.session.run(self.output_names, input_object)
    return ModelOutput(
      inputs = input_object,
      outputs = out,
    )

  def export(self, format: str, *a, **b):
    raise IllegalFormatError("ONNXRtModel cannot be exported")


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
  
class SklearnModel(FrameworkAgnosticModel):
  @isthere("sklearn", soft = False)
  def __init__(self, m1, m2):
    if not "sklearn" in str(type(m1)):
      raise IllegalFormatError
    if m2 != None and not callable(m2):
      # this is the processing logic for the incoming data this has to be a some kind of callable,
      # because user can define whatever they want and it removes any onus on NBX to ensure that
      # processing works, halting problem is no joke!
      IllegalFormatError(f"Second input must be a callable, got: {type(m2)}")
    
    self._model = m1
    self._logic = m2 # this is not used anywhere

  def forward(self, input_object: SklearnInput) -> ModelOutput:
    model_input = input_object.get("inputs", None)
    method = input_object.get("method", None)
    extra_args = input_object.get("kwargs", {})

    if "sklearn.neighbors.NearestNeighbors" in str(type(self.model_or_model_url)):
      method = getattr(self.model_or_model_url, "kneighbors") if method == None else getattr(self.model_or_model_url, method)
      out = method(model_input, **extra_args)
    elif "sklearn.cluster" in str(type(self.model_or_model_url)):
      if any(
        x in str(type(self.model_or_model_url)) for x in ["AgglomerativeClustering", "DBSCAN", "OPTICS", "SpectralClustering"]
      ):
        method = getattr(self.model_or_model_url, "fit_predict")
        out = method(model_input)
    else:
      try:
        method = getattr(self.model_or_model_url, "predict") if method == None else getattr(self.model_or_model_url, method)
        out = method(model_input)
      except Exception as e:
        print("[ERROR] Model Prediction Function is not yet registered " + e)
    
    return ModelOutput(
      inputs = model_input,
      outputs = out,
    )

  @isthere("skl2onnx", soft = False)
  def export_to_onnx(self, model, args, input_names, input_shapes, export_model_path, opset_version=None, **kwargs):
    from skl2onnx import convert_sklearn
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

    # the args need to be converted to proper formatted data types
    initial_types = []
    for name, shape, tensor in zip(input_names, input_shapes, args):
      shape = list(shape)
      shape[0] = None # batching requires the first dimension to be None
      initial_types.append((name, __NP_DTYPE_TO_SKL_DTYPE[str(tensor.dtype)](shape)))

    onx = convert_sklearn(model, initial_types=initial_types, target_opset=opset_version)

    with open(export_model_path, "wb") as f:
      f.write(onx.SerializeToString())

  def export_to_pkl(self, model, export_model_path, **kwargs):
    import joblib

    # sklearn models are pure python methods (though underlying contains bindings to C++)
    # and so we can use joblib for this
    # we use the joblib instead of pickle
    with open(export_model_path, "wb") as f:
      joblib.dump(model, f)

  def export(self, format, *a, **b) -> None:
    if format == "onnx":
      self.export_to_onnx(*a, **b)
    elif format == "pkl":
      self.export_to_pkl(*a, **b)
    else:
      raise IllegalFormatError(f"Unsupported export format for torch Module: {format}")
