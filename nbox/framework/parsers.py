# import os
# import numpy as np
# from PIL import Image

# from .. import utils

# import logging
# ()


# # ----- parsers
# # These objects are mux, they consume and streamline the output
# # Don't know what mux are? Study electronics.


# class BaseParser:
#   """This is the base parser class which has the required methods to process
#   the three kind of structures:

#   #. None
#   #. list: ``process_list()``
#   #. dict: ``process_dict()``

#   and a function dedicated to processing the primitives

#   #. ``process_primitive()``

#   These functions are called from the ``__call__`` method.
#   """

#   def __init__(self):
#     pass

#   def process_primitive(self):
#     raise NotImplementedError

#   def process_dict(self):
#     raise NotImplementedError

#   def process_list(self):
#     raise NotImplementedError

#   def __call__(self, input_object):
#     if isinstance(input_object, list):
#       logger.debug(f" - {self.__class__.__name__} - (A1)")
#       out = self.process_list(input_object)
#     elif isinstance(input_object, dict):
#       logger.debug(f" - {self.__class__.__name__} - (A2)")
#       out = self.process_dict(input_object)
#     else:
#       logger.debug(f" - {self.__class__.__name__} - (A3)")
#       out = self.process_primitive(input_object)

#     # apply self.post_proc_fn if it exists
#     if self.post_proc_fn:
#       # either you are going to get a dict or you are going to get a np.ndarray
#       if isinstance(out, dict):
#         if isinstance(next(iter(out.values())), dict):
#           # dict of dicts kinda thingy (say when two inputs are given)
#           out = {k: {_k: self.post_proc_fn(_v) for _k, _v in v.items()} for k, v in out.items()}
#         else:
#           out = {k: self.post_proc_fn(v) for k, v in out.items()}
#       else:
#         out = self.post_proc_fn(out)
#     return out


# # ----- parsers for each category


# class ImageParser(BaseParser):
#   def __init__(self, post_proc_fn=None, cloud_infer=False, **kwargs):
#     """single unified Image parser that consumes different types of data and returns a processed numpy array

#     Args:
#       post_proc_fn (callable, optional): post processing function, this takes in the torch tensor and performs
#         operation
#       cloud_infer (bool, optional): whether the input is from cloud inference
#       kwargs (dict, optional): keyword arguments to store here
#     """
#     super().__init__()
#     self.post_proc_fn = post_proc_fn
#     self.cloud_infer = cloud_infer
#     for k, v in kwargs.items():
#       setattr(self, k, v)

#   # common operations
#   # if not cloud_infer and input is int then rescale it 0, 1
#   def rescale(self, x: np.ndarray):
#     if not self.cloud_infer and "int" in str(x.dtype):
#       x = x / 255
#     return x

#   def rearrange(self, x: np.ndarray):
#     if len(x.shape) == 3 and x.shape[0] != 3:
#       return x.transpose(2, 0, 1)
#     elif len(x.shape) == 4 and x.shape[1] != 3:
#       return x.transpose(0, 3, 1, 2)
#     return x

#   def process_primitive(self, x, target_shape=None):
#     """primitive can be string, array, Image"""
#     if isinstance(x, np.ndarray):
#       logger.debug(" - ImageParser - (C2) np.ndarray")
#       # if shape == 3, unsqueeze it, numpy arrays cannot be reshaped
#       out = x[None, ...] if len(x.shape) == 3 else x
#     elif isinstance(x, Image.Image):
#       logger.debug(" - ImageParser - (C1) Image.Image object")
#       img = x
#     elif isinstance(x, str):
#       if os.path.isfile(x):
#         logger.debug(" - ImageParser - (C3) string - file")
#         img = Image.open(x)
#       elif x.startswith("http"):
#         logger.debug(" - ImageParser - (C4) string - url")
#         img = utils.get_image(x)
#       else:
#         try:
#           # probably base64
#           from io import BytesIO
#           import base64

#           logger.debug(" - ImageParser - (C5) string - base64")
#           img = Image.open(BytesIO(base64.b64decode(x)))
#         except:
#           raise Exception("Unable to parse string as Image")
#     else:
#       raise ValueError("Unknown primitive type: {}".format(type(x)))

#     if not isinstance(x, np.ndarray):
#       img = img.convert("RGB")

#       # this checks when only the primitive is sent directly but there is just one template
#       # so just use the only shape it has
#       if target_shape is None and hasattr(self, "templates") and len(self.templates) == 1:
#         target_shape = self.templates[next(iter(self.templates.keys()))]
#         target_shape = target_shape[-2:][::-1] # [h,w] -> [w,h]

#       # if a certain target shape is given, then resize it to that shape
#       if target_shape is not None:
#         target_shape = target_shape
#         img = img.resize(target_shape)

#       out = self.process_primitive(np.array(img))

#     # finally perform rearrange and rescale
#     return self.rescale(self.rearrange(out))

#   def process_dict(self, input_object, r_depth=0):
#     """takes in a dict, check if values are list, if list send to process_list
#     else process_primitive"""
#     out = {}

#     # if hasattar "templates" then the keys in input object should match
#     if hasattr(self, "templates"):
#       assert set(input_object.keys()) == set(
#         self.templates.keys()
#       ), f"input object keys do not match templates: {set(input_object.keys()) - set(self.templates.keys())}"

#     for k, v in input_object.items():
#       # if templates are given and the ket is same, then load that
#       target_shape = None
#       if hasattr(self, "templates"):
#         target_shape = self.templates[k]
#         target_shape = target_shape[-2:][::-1] # [h,w] -> [w,h]

#       # call the underlying object (structure or primitive)
#       if isinstance(v, list):
#         out[k] = self.process_list(v, target_shape, r_depth=r_depth + 1)
#       elif isinstance(v, dict):
#         out[k] = self.process_dict(v, r_depth=r_depth + 1)
#       else:
#         out[k] = self.process_primitive(v, target_shape)
#     return out

#   def process_list(self, input_object, target_shape=None, r_depth=0):
#     """takes in a list. This function is very tricky because the input
#     can be a list or a p-list, so we first check if input is not a string, Image or dict"""

#     # r_depth is the depth in the current reccursion, possible depths
#     # r_depth=0 -> [URL, URL]
#     # r_depth=1 -> {k:[URL, URL], l:[URL, URL]}
#     # r_depth=2 -> [{k:[URL, URL], l:[URL, URL]}, {k:[URL, URL], l:[URL, URL]}]}]
#     # thus if r_depth > 3, raise error
#     if r_depth >= 3:
#       raise RecursionError("Cannot go deeper with a list input")

#     if r_depth == 0 and hasattr(self, "templates") and len(self.templates) > 1:
#       raise ValueError(f"Template has more than 1 input, please input a dict with keys: {tuple(self.templates.keys())}")

#     if isinstance(input_object[0], (str, Image.Image)):
#       logger.debug(" - ImageParser - (B1) list - (str, Image.Image)")
#       out = [self.process_primitive(x, target_shape) for x in input_object]
#       return np.vstack(out)
#     elif isinstance(input_object[0], dict):
#       logger.debug(" - ImageParser - (B2) list - (dict)")
#       assert all([set(input_object[0].keys()) == set(i.keys()) for i in input_object]), "All keys must be same in all dicts in list"
#       out = [self.process_dict(x) for x in input_object]
#       out = {k: np.vstack([x[k] for x in out]) for k in out[0].keys()}
#       return out
#     else:
#       # check if this is a list of lists or np.ndarrays
#       if isinstance(input_object[0], list):
#         logger.debug(" - ImageParser - (B3) list - (list)")
#         # convert input_object to a np.array and check shapes - used in nbox-dply
#         out = np.array(input_object)
#         if len(out.shape) == 3:
#           out = out[None, ...]
#         return out
#       else:
#         logger.debug(" - ImageParser - (B4) list - (primitive)")
#         out = [self.process_primitive(x, target_shape) for x in input_object]
#         return np.vstack(out)


# class TextParser(BaseParser):
#   def __init__(self, tokenizer, max_len=None, **kwargs):
#     """Unified Text parsing engine, returns tokenized dictionaries

#     Args:
#       tokenizer (Tokenizer): tokenizer object for the text
#       max_len (int): maximum length of the text
#     """
#     super().__init__()
#     # tokenizer is supposed to be AutoTokenizer object, check that
#     self.tokenizer = tokenizer
#     self.max_len = max_len
#     for k, v in kwargs.items():
#       setattr(self, k, v)

#   def process_primitive(self, x):
#     # in case of text this is quite simple because only primitive is strings
#     if isinstance(x, str):
#       logger.debug(" - TextParser - (C1) string")
#       return {
#         k: np.array(v)[None, ...]
#         for k, v in self.tokenizer(
#           text=x,
#           add_special_tokens=True,
#           max_length=self.max_len,
#           padding="max_length" if self.max_len is not None else False,
#         ).items()
#       }
#     elif isinstance(x, np.ndarray):
#       logger.debug(" - TextParser - (C2) ndarray")
#       return x
#     else:
#       raise ValueError(f"Unsupported type for TextParser: {type(x)}")

#   def process_dict(self, input_object):
#     """takes in a dict and for each key's type call that method"""
#     out = {}
#     for k, v in input_object.items():
#       if isinstance(v, list):
#         out[k] = self.process_list(v)
#       elif isinstance(v, dict):
#         out[k] = self.process_dict(v)
#       else:
#         out[k] = self.process_primitive(v)
#     return out

#   def process_list(self, input_object):
#     """takes in and tokenises the strings"""
#     assert all([isinstance(x, str) for x in input_object]), "TextParser - (B1) input must be list of strings"
#     return {k: np.array(v) for k, v in self.tokenizer(input_object, padding="longest").items()}

# ssympple parsing

class Mux():
  @staticmethod
  def process_list(x):
    # type checking/
    t0 = type(x[0])
    if t0 == list:
      raise ValueError("Mux does not support nested lists")
    if any([type(x_) != t0 for x_ in x]):
      raise ValueError("Mux does not support mixed types")
    
    # logic/
    if isinstance(t0, dict):
      x = {k: Mux.process_list([x_[k] for x_ in x]) for k in x[0].keys()}
    else:
      x = Mux.primitive(x)
    
    return x
  
  @staticmethod
  def process_dict(x):
    for k, v in x.items():
      if isinstance(v, dict):
        x[k] = Mux.process_dict(v)
      elif isinstance(v, list):
        x[k] = Mux.process_list(v)
      else:
        x[k] = Mux.primitive(v)
    return x

  @staticmethod
  def parse(x, *a, **b):
    if isinstance(x, dict):
      return Mux.process_dict(x, *a, **b)
    elif isinstance(x, list):
      return Mux.process_list(x, *a, **b)
    else:
      return Mux.primitive(x, *a, **b)

  def primitive(x):
    pass
