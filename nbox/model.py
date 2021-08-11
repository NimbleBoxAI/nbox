import os
import torch
import numpy as np
from PIL import Image

from nbox import processing
from nbox.utils import info, is_available

# ----- parsers
# These objects are mux, they consume and streamline the output
# Don't know what mux are? Study electronics.


class ImageParser:
    """single unified Image parser that returns PIL.Image objects by consuming multiple differnet data-types"""
    def handle_string(self, x):
        if os.path.exists(x):
            info(" - ImageParser - (A)")
            return [Image.open(x)]
        else:
            info(" - ImageParser - (B)")
            raise ValueError("Cannot process string that is not Image path")

    def handle_numpy(self, obj):
        if obj.dtype == np.float32 or obj.dtype == np.float64:
            info(" - ImageParser - (C)")
            obj *= 122.5
            obj += 122.5
        info(" - ImageParser - (C2)")
        if obj.dtype != np.uint8:
            obj = obj.astype(np.uint8)
        return [Image.fromarray(obj)]

    def handle_torch_tensor(self, obj):
        if obj.dtype == torch.float:
            info(" - ImageParser - (C)")
            obj *= 122.5
            obj += 122.5
            obj = obj.numpy()
        else:
            raise ValueError(f"Incorrect datatype for torch.tensor: {obj.dtype}")
        info(" - ImageParser - (C2)")
        if obj.dtype != np.uint8:
            obj = obj.astype(np.uint8)
        return [Image.fromarray(obj)]

    def __call__(self, x):
        if isinstance(x, str):
            proc_fn = self.handle_string
        elif isinstance(x, np.ndarray):
            proc_fn = self.handle_numpy
        elif isinstance(x, Image.Image):
            info(" - ImageParser - (D)")
            proc_fn = lambda x: [x]
        else:
            info(" - ImageParser - (E)")
            raise ValueError(f"Cannot process item of dtype: {type(x)}")
        return proc_fn(x)


class TextParser:
    """Unified Text parsing engine, returns a list of dictionaries"""
    def handle_string(self, x, tokenizer):
        info(" - TextParser - (A)")
        return {k: v.numpy() for k, v in tokenizer(x, return_tensors="pt").items()}

    def handle_numpy(self, x, tokenizer=None):
        info(" - TextParser - (B)")
        if x.dtype != np.int32 and x.dtype != np.int64:
            raise ValueError(f"Incorrect datatype for np.array: {x.dtype} | {x.dtype == np.int64}")
        return {"input_ids": x}

    def handle_dict(self, x, tokenizer=None):
        _x = {}
        info(" - TextParser - (C)")
        for k, v in x.items():
            if isinstance(v, (list, tuple)) and isinstance(v[0], (list, tuple)):
                # list of lists -> batch processing
                info(" - TextParser - (C1)")
                seqlen = len(v[0])
                assert len(v) * seqlen == sum([len(x) for x in v]), "Inconsistent values in list sequences"
                _x[k] = np.array(v).astype(np.int32)
            elif isinstance(v, (list, tuple)) and isinstance(v[0], int):
                info(" - TextParser - (C2)")
                _x[k] = np.array(v).astype(np.int32)
            else:
                raise ValueError("Cannot parse dict items")
        return _x

    def handle_list_tuples(self, x, tokenizer=None):
        info(" - TextParser - (D)")
        if isinstance(x[0], int):
            info(" - TextParser - (D1)")
        elif isinstance(x[0], list):
            info(" - TextParser - (D2)")
            seqlen = len(x[0])
            assert len(x) * seqlen == sum([len(y) for y in x]), "Inconsistent values in list sequences"
        elif isinstance(x[0], str):
            info(" - TextParser - (D3)")
            if tokenizer == None:
                raise ValueError("tokenizer cannot be None when string input")
            tokens = tokenizer(x, padding="longest", return_tensors="pt")
            return {k: v.numpy().astype(np.int32) for k, v in tokens.items()}
        else:
            raise ValueError(f"Cannot parse list of item: {type(x[0])}")
        return {"input_ids": np.array(x).astype(np.int32)}

    def handle_torch_tensor(self, x, tokenizer = None):
        info(" - Text Parser - (F)")
        assert x.dtype == torch.long, f"Incorrect datatype for torch.tensor: {x.dtype}"
        return {"input_ids": x}

    def __call__(self, x, tokenizer=None):
        if isinstance(x, str):
            if tokenizer is None:
                raise ValueError("tokenizer cannot be None when string input")
            proc_fn = self.handle_string
        elif isinstance(x, np.ndarray):
            proc_fn = self.handle_numpy
        elif isinstance(x, dict):
            proc_fn = self.handle_dict
        elif isinstance(x, (list, tuple)):
            proc_fn = self.handle_list_tuples
        elif is_available("torch") and isinstance(x, torch.Tensor):
            proc_fn = self.handle_torch_tensor
        else:
            info(" - ImageParser - (E)")
            raise ValueError(f"Cannot process item of dtype: {type(x)}")
        return proc_fn(x, tokenizer)


class Model:
    """Nbox.Model class designed for inference"""

    def __init__(self, model: torch.nn.Module, category, tokenizer=None):
        self.model = model
        self.category = category

        # initialise all the parsers, like WTH, how bad would it be
        self.image_parser = ImageParser()
        self.text_parser = TextParser()

        if self.category not in ["image", "text"]:
            raise ValueError(f"Category: {self.category} is not supported yet. Raise a PR!")

        if self.category == "text":
            assert tokenizer != None, "tokenizer cannot be none for a text model!"
            self.tokenizer = tokenizer

    def get_model(self):
        return self.model

    def __call__(self, input_object):
        """This is the most important part of this codebase. The `input_object` can be anything from
        a tensor, an image file, filepath as string, string to process as NLP model. This `__call__`
        should understand the different usecases and manage accordingly.

        The current idea is that what ever the input, based on the category (image, text, audio, smell)
        it will be parsed through dedicated parsers that can make ingest anything.

        The entire purpose of this package is to make inference chill."""

        if self.category == "image":
            # perform parsing for images
            if isinstance(input_object, (list, tuple)):
                _t = []
                for item in input_object:
                    pil_img = self.image_parser(item)[0]
                    _t.append(processing.totensor(pil_img))
                input_tensor = torch.cat(_t, axis=0)
            else:
                pil_img = self.image_parser(input_object)[0]
                input_tensor = processing.totensor(pil_img)
            out = self.model(input_tensor)  # call the model

        elif self.category == "text":
            # perform parsing for text and pass to the model
            input_dict = self.text_parser(input_object, self.tokenizer)
            input_dict = {k: torch.from_numpy(v) for k, v in input_dict.items()}
            out = self.model(**input_dict)

        return out

    def eval(self):
        self.model.eval()

    def train(self):
        self.model.train()

    def deploy(self, nbx_api_key, machine_id):
        # this is a part of one-click to NBX
        raise NotImplementedError()

    def export(self, folder_path):
        # creates a FastAPI / Flask folder with all the things required to serve this model
        raise NotImplementedError()
