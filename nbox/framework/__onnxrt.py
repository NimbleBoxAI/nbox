import onnxruntime

from nbox.framework.common import IllegalFormatError


def load_model(model, ):
  if not isinstance(model, onnxruntime.InferenceSession):
    raise IllegalFormatError

  logger.info(f"Trying to load from onnx model: {model}")
  __framework = "onnx"

  # we have to create templates using the nbox_meta
  templates = None
  if nbox_meta is not None:
    all_inputs = nbox_meta["metadata"]["inputs"]
    templates = {}
    for node, meta in all_inputs.items():
      templates[node] = [int(x["size"]) for x in meta["tensorShape"]["dim"]]

  image_parser = ImageParser(post_proc_fn=lambda x: x.astype(np.float32), templates = templates)
  text_parser = TextParser(tokenizer=tokenizer, post_proc_fn=lambda x: x.astype(np.int32))

  self.session = model
  self.input_names = [x.name for x in self.session.get_inputs()]
  self.output_names = [x.name for x in self.session.get_outputs()]

  logger.info(f"Inputs: {self.input_names}")
  logger.info(f"Outputs: {self.output_names}")

  return ModelMeta(
    framework = "onnxruntime",
  )

def forward_pass(meta: ModelMeta):
  if set(model_input.keys()) != set(self.input_names):
    diff = set(model_input.keys()) - set(self.input_names)
    return f"model_input keys do not match input_name: {diff}"
  out = self.model_or_model_url.run(self.output_names, model_input)
  return out

class ONNXRtMixin:
  @isthere("onnxruntime", soft = False)
  def load_model(*a, **b):
    return load_model(*a, **b)

  @isthere("onnxruntime", soft = False)
  def forward_pass(*a, **b):
    return forward_pass(*a, **b)

__all__ = []