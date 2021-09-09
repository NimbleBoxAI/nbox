import os
import torch
import numpy as np
from PIL import Image

from nbox import utils


# ----- parsers
# These objects are mux, they consume and streamline the output
# Don't know what mux are? Study electronics.


class ImageParser:
    """single unified Image parser that consumes different types of data
    and returns a processed numpy array"""

    def __init__(self, pre_proc_fn=None, cloud_infer=False):
        self.pre_proc_fn = pre_proc_fn
        self.cloud_infer = cloud_infer

    # common operations
    # if the input is torch.int then rescale it -1, 1
    def rescale(self, x):
        if not self.cloud_infer:
            if utils.is_available("torch") and isinstance(x, torch.Tensor) and "int" in str(x.dtype):
                return (x - 122.5) / 122.5
        return x

    def rearrange(self, x):
        if len(x.shape) == 3 and x.shape[0] != 3:
            return x.permute(2, 0, 1)
        elif len(x.shape) == 4 and x.shape[1] != 3:
            return x.permute(0, 3, 1, 2)
        return x

    # ---- handler functions
    def handle_string(self, x: str):
        if os.path.exists(x):
            utils.info(" - ImageParser - (A1)")
            out = np.array(Image.open(x))
            return out
        elif x.startswith("http:") or x.startswith("https:"):
            utils.info(" - ImageParser - (A2)")
            out = np.array(utils.get_image(x))
            return out
        else:
            utils.info(" - ImageParser - (A)")
            raise ValueError("Cannot process string that is not Image path")

    def handle_list(self, x: list):
        utils.info(" - ImageParser - (B)")
        return [self(i) for i in x]

    def handle_dictionary(self, x: dict):
        utils.info(" - ImageParser - (C)")
        return {k: self(v) for k, v in x.items()}

    def handle_pil_image(self, x: Image):
        utils.info(" - ImageParser - (D)")
        out = self(np.array(x))
        return out

    def __call__(self, x):
        if utils.is_available("torch") and isinstance(x, torch.Tensor):
            x = x
            x = self.pre_proc_fn(x) if self.pre_proc_fn is not None else x
            out = self.rearrange(self.rescale(x.unsqueeze(0)))
        elif isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
            x = self.pre_proc_fn(x) if self.pre_proc_fn is not None else x
            out = self.rearrange(self.rescale(x.unsqueeze(0)))
        elif isinstance(x, str):
            x = torch.from_numpy(self.handle_string(x))
            x = self.pre_proc_fn(x) if self.pre_proc_fn is not None else x
            out = self.rearrange(self.rescale(x.unsqueeze(0)))
        elif isinstance(x, list):
            out = self.handle_list(x)

            # if out is list of dicts then create a dict with concatenated tensors
            if isinstance(out[0], dict):
                # assert all keys are same in the list
                assert all([set(out[0].keys()) == set(i.keys()) for i in out]), "All keys must be same in all dicts in list"
                out = {k: torch.cat([i[k] for i in out]) for k in out[0].keys()}
            elif isinstance(out[0], torch.Tensor):
                out = torch.cat(out)
            else:
                raise ValueError("Unsupported type: {}".format(type(out[0])))

        elif isinstance(x, dict):
            x = self.handle_dictionary(x)
            x = {k: self.pre_proc_fn(v) if self.pre_proc_fn is not None else v for k, v in x.items()}
            out = x
        elif isinstance(x, Image.Image):
            x = self.handle_pil_image(x)
            out = self.rearrange(self.rescale(x))
        else:
            utils.info(" - ImageParser - (D)")
            raise ValueError(f"Cannot process item of dtype: {type(x)}")
        return out


class TextParser:
    """Unified Text parsing engine, returns a list of dictionaries"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def handle_string(self, x):
        utils.info(" - TextParser - (A)")
        return {k: v.numpy() for k, v in self.tokenizer(x, return_tensors="pt").items()}

    def handle_numpy(self, x):
        utils.info(" - TextParser - (B)")
        if x.dtype != np.int32 and x.dtype != np.int64:
            raise ValueError(f"Incorrect datatype for np.array: {x.dtype} | {x.dtype == np.int64}")
        return {"input_ids": x}

    def handle_dict(self, x):
        _x = {}
        utils.info(" - TextParser - (C)")
        for k, v in x.items():
            if isinstance(v, (list, tuple)) and isinstance(v[0], (list, tuple)):
                # list of lists -> batch processing
                utils.info(" - TextParser - (C1)")
                seqlen = len(v[0])
                assert len(v) * seqlen == sum([len(x) for x in v]), "Inconsistent values in list sequences"
                _x[k] = np.array(v).astype(np.int32)
            elif isinstance(v, (list, tuple)) and isinstance(v[0], int):
                utils.info(" - TextParser - (C2)")
                _x[k] = np.array(v).astype(np.int32)
            else:
                raise ValueError("Cannot parse dict items")
        return _x

    def handle_list_tuples(self, x):
        utils.info(" - TextParser - (D)")
        if isinstance(x[0], int):
            utils.info(" - TextParser - (D1)")
        elif isinstance(x[0], list):
            utils.info(" - TextParser - (D2)")
            seqlen = len(x[0])
            assert len(x) * seqlen == sum([len(y) for y in x]), "Inconsistent values in list sequences"
        elif isinstance(x[0], str):
            utils.info(" - TextParser - (D3)")
            if self.tokenizer == None:
                raise ValueError("self.tokenizer cannot be None when string input")
            tokens = self.tokenizer(x, padding="longest", return_tensors="pt")
            return {k: v.numpy().astype(np.int32) for k, v in tokens.items()}
        else:
            raise ValueError(f"Cannot parse list of item: {type(x[0])}")
        return {"input_ids": np.array(x).astype(np.int32)}

    def handle_torch_tensor(self, x):
        utils.info(" - Text Parser - (F)")
        assert x.dtype == torch.long, f"Incorrect datatype for torch.tensor: {x.dtype}"
        return {"input_ids": x}

    def __call__(self, x):
        if isinstance(x, str):
            proc_fn = self.handle_string
        elif isinstance(x, np.ndarray):
            proc_fn = self.handle_numpy
        elif isinstance(x, dict):
            proc_fn = self.handle_dict
        elif isinstance(x, (list, tuple)):
            proc_fn = self.handle_list_tuples
        elif utils.is_available("torch") and isinstance(x, torch.Tensor):
            proc_fn = self.handle_torch_tensor
        else:
            utils.info(" - ImageParser - (E)")
            raise ValueError(f"Cannot process item of dtype: {type(x)}")
        return proc_fn(x)
