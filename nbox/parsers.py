import os
import torch
import numpy as np
from PIL import Image

from nbox import utils


# ----- parsers
# These objects are mux, they consume and streamline the output
# Don't know what mux are? Study electronics.


class BaseParser:
    """This is the base parser class which has the required methods to process
    the three kind of structures:
    * None
    * list: process_list()
    * dict: process_dict()

    and a function dedicated to processing the primitives
    * process_primitive()

    These functions are called from the __call__ method.

    read more about parsers in docs:
        https://docs.nimblebox.ai/nbox/python-api/inference-parsing
    """

    def __init__(self):
        pass

    def process_primitive(self):
        raise NotImplementedError

    def process_dict(self):
        raise NotImplementedError

    def process_list(self):
        raise NotImplementedError

    def __call__(self, input_object):
        if isinstance(input_object, list):
            utils.info(f" - {self.__class__.__name__} - (A1)")
            out = self.process_list(input_object)
        elif isinstance(input_object, dict):
            utils.info(f" - {self.__class__.__name__} - (A2)")
            out = self.process_dict(input_object)
        else:
            utils.info(f" - {self.__class__.__name__} - (A3)")
            out = self.process_primitive(input_object)

        # apply self.post_proc_fn if it exists
        if self.post_proc_fn:
            # either you are going to get a dict or you are going to get a np.ndarray
            if isinstance(out, dict):
                if isinstance(next(iter(out.values())), dict):
                    # dict of dicts kinda thingy (say when two inputs are given)
                    out = {k: {_k: self.post_proc_fn(_v) for _k, _v in v.items()} for k, v in out.items()}
                else:
                    out = {k: self.post_proc_fn(v) for k, v in out.items()}
            else:
                out = self.post_proc_fn(out)
        return out


# ----- parsers for each category


class ImageParser(BaseParser):
    """single unified Image parser that consumes different types of data
    and returns a processed numpy array"""

    def __init__(self, post_proc_fn=None, cloud_infer=False):
        super().__init__()
        self.post_proc_fn = post_proc_fn
        self.cloud_infer = cloud_infer

    # common operations
    # if not cloud_infer and input is int then rescale it -1, 1
    def rescale(self, x: np.ndarray):
        if not self.cloud_infer and "int" in str(x.dtype):
            return (x - 122.5) / 122.5
        return x

    def rearrange(self, x: np.ndarray):
        if len(x.shape) == 3 and x.shape[0] != 3:
            return x.transpose(2, 0, 1)
        elif len(x.shape) == 4 and x.shape[1] != 3:
            return x.transpose(0, 3, 1, 2)
        return x

    def process_primitive(self, x):
        """primitive can be string, array, Image"""
        if isinstance(x, Image.Image):
            utils.info(" - ImageParser - (C1) Image.Image object")
            out = self.process_primitive(np.array(x))
        elif isinstance(x, np.ndarray):
            utils.info(" - ImageParser - (C2) np.ndarray")
            # if shape == 3, unsqueeze it
            out = x[None, ...] if len(x.shape) == 3 else x
        elif isinstance(x, str):
            if os.path.isfile(x):
                utils.info(" - ImageParser - (C3) string - file")
                out = self.process_primitive(np.array(Image.open(x)))
            elif x.startswith("http"):
                utils.info(" - ImageParser - (C4) string - url")
                out = self.process_primitive(np.array(utils.get_image(x)))
            else:
                try:
                    # probably base64
                    from io import BytesIO
                    import base64

                    utils.info(" - ImageParser - (C5) string - base64")
                    out = self.process_primitive(np.array(Image.open(BytesIO(base64.b64decode(x)))))
                except:
                    raise Exception("Unable to parse string as Image")
        else:
            raise ValueError("Unknown primitive type: {}".format(type(x)))

        # finally perform rearrange and rescale
        return self.rescale(self.rearrange(out))

    def process_dict(self, input_object):
        """takes in a dict, check if values are list, if list send to process_list
        else process_primitive"""
        out = {}
        for k, v in input_object.items():
            if isinstance(v, list):
                out[k] = self.process_list(v)
            elif isinstance(v, dict):
                out[k] = self.process_dict(v)
            else:
                out[k] = self.process_primitive(v)
        return out

    def process_list(self, input_object):
        """takes in a list. This function is very tricky because the input
        can be a list or a p-list, so we first check if input is not a string, Image or dict"""
        if isinstance(input_object[0], (str, Image.Image)):
            utils.info(" - ImageParser - (B1) list - (str, Image.Image)")
            out = [self.process_primitive(x) for x in input_object]
            return np.vstack(out)
        if isinstance(input_object[0], dict):
            utils.info(" - ImageParser - (B2) list - (dict)")
            assert all([set(input_object[0].keys()) == set(i.keys()) for i in input_object]), "All keys must be same in all dicts in list"
            out = [self.process_dict(x) for x in input_object]
            out = {k: np.vstack([x[k] for x in out]) for k in out[0].keys()}
            return out
        else:
            # check if this is a list of lists or np.ndarrays or torch.Tensors
            if isinstance(input_object[0], list):
                utils.info(" - ImageParser - (B3) list - (list)")
                # convert input_object to a np.array and check shapes
                out = np.array(input_object)
                if len(out.shape) == 3:
                    out = out[None, ...]
                return out
            else:
                utils.info(" - ImageParser - (B4) list - (primitive)")
                out = [self.process_primitive(x) for x in input_object]
                return np.vstack(out)


class TextParser(BaseParser):
    """Unified Text parsing engine, returns tokenized dictionaries"""

    def __init__(self, tokenizer, max_len = None, post_proc_fn=None):
        super().__init__()
        # tokenizer is supposed to be AutoTokenizer object, check that
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.post_proc_fn = post_proc_fn

    def process_primitive(self, x):
        # in case of text this is quite simple because only primitive is strings
        assert isinstance(x, str), "TextParser - (C1) input must be string"
        return {
            k: np.array(v)[None, ...] for k, v in self.tokenizer(
                x,
                max_length = self.max_len,
                padding="max_length"
            ).items()
        }

    def process_dict(self, input_object):
        """takes in a dict and for each key's type call that method"""
        out = {}
        for k, v in input_object.items():
            if isinstance(v, list):
                out[k] = self.process_list(v)
            elif isinstance(v, dict):
                out[k] = self.process_dict(v)
            else:
                out[k] = self.process_primitive(v)
        return out

    def process_list(self, input_object):
        """takes in and tokenises the strings"""
        assert all([isinstance(x, str) for x in input_object]), "TextParser - (B1) input must be list of strings"
        return {k: np.array(v) for k, v in self.tokenizer(input_object, padding="longest").items()}


# the class ImageParser below is only added here for reference. Itis not used in the code
# this was user till __version__ == 0.1.7

# class ImageParser:
#     """single unified Image parser that consumes different types of data
#     and returns a processed numpy array"""
#
#     def __init__(self, pre_proc_fn=None, cloud_infer=False):
#         self.pre_proc_fn = pre_proc_fn
#         self.cloud_infer = cloud_infer
#
#     # common operations
#     # if the input is torch.int then rescale it -1, 1
#     def rescale(self, x):
#         if not self.cloud_infer:
#             if utils.is_available("torch") and isinstance(x, torch.Tensor) and "int" in str(x.dtype):
#                 return (x - 122.5) / 122.5
#         return x
#
#     def rearrange(self, x):
#         if len(x.shape) == 3 and x.shape[0] != 3:
#             return x.permute(2, 0, 1)
#         elif len(x.shape) == 4 and x.shape[1] != 3:
#             return x.permute(0, 3, 1, 2)
#         return x
#
#     # ---- handler functions
#     def handle_string(self, x: str):
#         if os.path.exists(x):
#             utils.info(" - ImageParser - (A1)")
#             out = np.array(Image.open(x))
#             return out
#         elif x.startswith("http:") or x.startswith("https:"):
#             utils.info(" - ImageParser - (A2)")
#             out = np.array(utils.get_image(x))
#             return out
#         else:
#             utils.info(" - ImageParser - (A)")
#             raise ValueError("Cannot process string that is not Image path")
#
#     def handle_list(self, x: list):
#         utils.info(" - ImageParser - (B)")
#         out = np.array(x)
#         if len(out.shape) == 3:
#             return self(out)
#         return [self(i) for i in x]
#
#     def handle_dictionary(self, x: dict):
#         utils.info(" - ImageParser - (C)")
#         return {k: self(v) for k, v in x.items()}
#
#     def handle_pil_image(self, x: Image):
#         utils.info(" - ImageParser - (D)")
#         out = self(np.array(x))
#         return out
#
#     def __call__(self, x):
#         print("--->>>>>", type(x))
#         if utils.is_available("torch") and isinstance(x, torch.Tensor):
#             x = x
#             x = self.pre_proc_fn(x) if self.pre_proc_fn is not None else x
#             out = self.rearrange(self.rescale(x.unsqueeze(0)))
#         elif isinstance(x, np.ndarray):
#             x = torch.from_numpy(x)
#             x = self.pre_proc_fn(x) if self.pre_proc_fn is not None else x
#             out = self.rearrange(self.rescale(x.unsqueeze(0)))
#         elif isinstance(x, str):
#             x = torch.from_numpy(self.handle_string(x))
#             x = self.pre_proc_fn(x) if self.pre_proc_fn is not None else x
#             out = self.rearrange(self.rescale(x.unsqueeze(0)))
#         elif isinstance(x, list):
#             out = self.handle_list(x)
#             # if out is list of dicts then create a dict with concatenated tensors
#             if isinstance(out[0], dict):
#                 # assert all keys are same in the list
#                 assert all([set(out[0].keys()) == set(i.keys()) for i in out]), "All keys must be same in all dicts in list"
#                 f_key = next(iter(x[0].keys()))
#                 if len(out[f_key]) > 1:
#                     out = {k: torch.cat([i[k] for i in out]) for k in out[0].keys()}
#             elif isinstance(out[0], torch.Tensor):
#                 if len(out[0]) > 1:
#                     out = torch.cat(out)
#             else:
#                 raise ValueError("Unsupported type: {}".format(type(out[0])))
#         elif isinstance(x, dict):
#             x = self.handle_dictionary(x)
#             x = {k: self.pre_proc_fn(v) if self.pre_proc_fn is not None else v for k, v in x.items()}
#             out = x
#         elif isinstance(x, Image.Image):
#             x = self.handle_pil_image(x)
#             out = self.rearrange(self.rescale(x))
#         else:
#             utils.info(" - ImageParser - (D)")
#             raise ValueError(f"Cannot process item of dtype: {type(x)}")
#         return out


# the class TextParser below is only added here for reference. It is not used in the code
# this was user till __version__ == 0.1.7

# class TextParser():
#     """Unified Text parsing engine, returns a list of dictionaries"""
#     def __init__(self, tokenizer):
#         self.tokenizer = tokenizer
#
#     def handle_string(self, x):
#         utils.info(" - TextParser - (A)")
#         return {k: v.numpy() for k, v in self.tokenizer(x, return_tensors="pt").items()}
#
#     def handle_numpy(self, x):
#         utils.info(" - TextParser - (B)")
#         if x.dtype != np.int32 and x.dtype != np.int64:
#             raise ValueError(f"Incorrect datatype for np.array: {x.dtype} | {x.dtype == np.int64}")
#         return {"input_ids": x}
#
#     def handle_dict(self, x):
#         _x = {}
#         utils.info(" - TextParser - (C)")
#         for k, v in x.items():
#             if isinstance(v, (list, tuple)) and isinstance(v[0], (list, tuple)):
#                 # list of lists -> batch processing
#                 utils.info(" - TextParser - (C1)")
#                 seqlen = len(v[0])
#                 assert len(v) * seqlen == sum([len(x) for x in v]), "Inconsistent values in list sequences"
#                 _x[k] = np.array(v).astype(np.int32)
#             elif isinstance(v, (list, tuple)) and isinstance(v[0], int):
#                 utils.info(" - TextParser - (C2)")
#                 _x[k] = np.array(v).astype(np.int32)
#             else:
#                 raise ValueError("Cannot parse dict items")
#         return _x
#
#     def handle_list_tuples(self, x):
#         utils.info(" - TextParser - (D)")
#         if isinstance(x[0], int):
#             utils.info(" - TextParser - (D1)")
#         elif isinstance(x[0], list):
#             utils.info(" - TextParser - (D2)")
#             seqlen = len(x[0])
#             assert len(x) * seqlen == sum([len(y) for y in x]), "Inconsistent values in list sequences"
#         elif isinstance(x[0], str):
#             utils.info(" - TextParser - (D3)")
#             if self.tokenizer == None:
#                 raise ValueError("self.tokenizer cannot be None when string input")
#             tokens = self.tokenizer(x, padding="longest", return_tensors="pt")
#             return {k: v.numpy().astype(np.int32) for k, v in tokens.items()}
#         else:
#             raise ValueError(f"Cannot parse list of item: {type(x[0])}")
#         return {"input_ids": np.array(x).astype(np.int32)}
#
#     def handle_torch_tensor(self, x):
#         utils.info(" - Text Parser - (F)")
#         assert x.dtype == torch.long, f"Incorrect datatype for torch.tensor: {x.dtype}"
#         return {"input_ids": x}
#
#     def __call__(self, x):
#         if isinstance(x, str):
#             proc_fn = self.handle_string
#         elif isinstance(x, np.ndarray):
#             proc_fn = self.handle_numpy
#         elif isinstance(x, dict):
#             proc_fn = self.handle_dict
#         elif isinstance(x, (list, tuple)):
#             proc_fn = self.handle_list_tuples
#         elif utils.is_available("torch") and isinstance(x, torch.Tensor):
#             proc_fn = self.handle_torch_tensor
#         else:
#             utils.info(" - ImageParser - (E)")
#             raise ValueError(f"Cannot process item of dtype: {type(x)}")
#         return proc_fn(x)
