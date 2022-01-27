from .on_functions import DBase

class IllegalFormatError(Exception):
  pass

class ModelMeta(DBase):
  __slots__ = [
    "framework", # :str: enum("nbx", "skl", "pkl", "pt")
    "forward_pass", # :callable:
    "model", # :Callable: this function is called with users input
    "model_extra", # :Dict: extra metadata for this model
    "nbox_meta", # :Dict: nbox_meta
    "parser", # :Callable: this function is called with users input
    "parser_extra", # :Dict: extra metadata for this parser
  ]

class FrameworkAgnosticModel:
  def __init__(self):
    pass

  def forward(self):
    raise NotImplementedError("User must implement forward()")

  def process_input(input_object):
    raise NotImplementedError("User must implement process_input()")

  def __call__(self, input_object):
    return self.forward(input_object)
