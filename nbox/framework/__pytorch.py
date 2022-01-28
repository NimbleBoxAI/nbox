# this file has the utilities and functions required for processing pytorch items
# such as conversion to ONNX, getting the metadata and so on.

from re import L
from typing import Dict
import torch

from logging import getLogger
logger = getLogger()

from .common import ModelOutput, IllegalFormatError, FrameworkAgnosticModel


def export_to_onnx(
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


def export_to_torchscript(model, args, export_model_path, **kwargs):
  traced_model = torch.jit.trace(model, args, check_tolerance=0.0001)
  torch.jit.save(traced_model, export_model_path)


def load_model(model, inputs):
  logger.info(f"Trying to load as torch model")
  if not isinstance(model, torch.nn.Module):
    raise IllegalFormatError
  
  if not (inputs == None or isinstance(inputs, dict)):
    raise ValueError(f"Inputs must be a None/dict, got: {type(inputs)}")

  def forward_pass(input_object):
    if isinstance(inputs, dict):
      if not isinstance(input_object, dict):
        raise ValueError(f"Inputs must be a dict, got: {type(input_object)}")
      if set(inputs.keys()) != set(input_object.keys()):
        raise ValueError(f"Inputs keys do not match: {inputs.keys()} != {input_object.keys()}")
      
      _input = {}
      for k, v in input_object.items():
        if inputs[k] != None and callable(inputs[k]):
          _input[k] = inputs[k](v)
        else:
          _input[k] = v
      input_object = _input

    with torch.no_grad():
      if isinstance(input_object, dict):
        return model(**{k: v.to("cpu") for k, v in input_object.items()})
      else:
        assert isinstance(input_object, torch.Tensor)
        return model(input_object)

  return ModelMeta(
    framework = "pytorch",
    forward_pass = forward_pass,
  )


class TorchMixin:
  def load_model(*a, **b):
    return load_model(*a, **b)

  def export_to_onnx(*a, **b):
    export_to_onnx(*a, **b)
  
  def export_to_torchscript(*a, **b):
    export_to_torchscript(*a, **b)


class TorchModel(FrameworkAgnosticModel):
  def __init__(self, m1, m2):
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

  def export(format, *a, **b) -> None:
    if format == "onnx":
      export_to_onnx(*a, **b)
    elif format == "torchscript":
      export_to_torchscript(*a, **b)
    else:
      raise IllegalFormatError(f"Unsupported export format for torch Module: {format}")
