# this file has the utilities and functions required for processing pytorch items
# such as conversion to ONNX, getting the metadata and so on.

import torch

from nbox.framework.common import IllegalFormatError

from .common import ModelMeta


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


def load_model(model, ):
  if not isinstance(model, torch.nn.Module):
    raise IllegalFormatError

  logger.info(f"Trying to load from torch model")
  __framework = "pytorch"
  assert category is not None, "Category for inputs must be provided, when loading model manually"

  model_key = model_key

  # initialise all the parsers
  image_parser = ImageParser(post_proc_fn=lambda x: torch.from_numpy(x).float())
  text_parser = TextParser(tokenizer=tokenizer, post_proc_fn=lambda x: torch.from_numpy(x).int())

  if isinstance(category, dict):
    assert all([v in ["image", "text", "tensor"] for v in category.values()])
  else:
    if category not in ["image", "text", "tensor"]:
      raise ValueError(f"Category: {category} is not supported yet. Raise a PR!")

  if category == "text":
    assert tokenizer != None, "tokenizer cannot be none for a text model!"

  return ModelMeta(
    framework = "pytorch",
  )

def forward_pass(meta: ModelMeta):
  with torch.no_grad():
    if isinstance(model_input, dict):
      model_input = {k: v.to(self.__device) for k, v in model_input.items()}
      out = self.model_or_model_url(**model_input)
    else:
      assert isinstance(model_input, torch.Tensor)
      model_input = model_input.to(self.__device)
      out = self.model_or_model_url(model_input)
  return out


class TorchMixin:
  def load_model(*a, **b):
    return load_model(*a, **b)

  def forward_pass(*a, **b):
    return forward_pass(*a, **b)

  def export_to_onnx(*a, **b):
    export_to_onnx(*a, **b)
  
  def export_to_torchscript(*a, **b):
    export_to_torchscript(*a, **b)

