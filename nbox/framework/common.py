from .on_functions import DBase

class IllegalFormatError(Exception):
  pass

class ModelMeta(DBase):
  __slots__ = [
    "framework", # :str: enum("nbx", "skl", "pkl", "pt")
    "model", # :Callable: this function is called with users input
    "model_extra", # :Dict: extra metadata for this model
    "nbox_meta", # :Dict: nbox_meta
    "parser", # :Callable: this function is called with users input
    "parser_extra", # :Dict: extra metadata for this parser
  ]

