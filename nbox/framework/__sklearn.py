import joblib
from nbox.framework.common import IllegalFormatError
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


def export_to_onnx(model, args, input_names, input_shapes, export_model_path, opset_version=None, **kwargs):
  # the args need to be converted to proper formatted data types
  initial_types = []
  for name, shape, tensor in zip(input_names, input_shapes, args):
    shape = list(shape)
    shape[0] = None # batching requires the first dimension to be None
    initial_types.append((name, __NP_DTYPE_TO_SKL_DTYPE[str(tensor.dtype)](shape)))

  onx = convert_sklearn(model, initial_types=initial_types, target_opset=opset_version)

  with open(export_model_path, "wb") as f:
    f.write(onx.SerializeToString())


def export_to_pkl(model, export_model_path, **kwargs):
  # sklearn models are pure python methods (though underlying contains bindings to C++)
  # and so we can use joblib for this
  # we use the joblib instead of pickle
  with open(export_model_path, "wb") as f:
    joblib.dump(model, f)




def load_model(model, ):
  if not "sklearn" in str(type(model_or_model_url)):
    raise IllegalFormatError

  return ModelMeta(
    framework = "sklearn",
  )


def forward_pass(meta: ModelMeta):
  if "sklearn.neighbors.NearestNeighbors" in str(type(self.model_or_model_url)):
    method = getattr(self.model_or_model_url, "kneighbors") if method == None else getattr(self.model_or_model_url, method)
    out = method(model_input, **sklearn_args)
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

class SklearnMixin:
  @isthere("torch", soft = False)
  def load_model(*a, **b):
    return load_model(*a, **b)

  @isthere("torch", soft = False)
  def forward_pass(*a, **b):
    return forward_pass(*a, **b)

  @isthere("torch", soft = False)
  def export_to_onnx(*a, **b):
    export_to_onnx(*a, **b)
  
  @isthere("torch", soft = False)
  def export_to_pkl(*a, **b):
    export_to_pkl(*a, **b)


__all__ = ["export_to_onnx", "export_to_pkl"]
