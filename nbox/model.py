import os
import torch
import numpy as np
from PIL import Image

from nbox import processing
from nbox.utils import info


class ImageParser:
    # single unified Image parser that returns PIL.Image objects by
    # consuming multiple differnet data-types
    def handle_string(self, x):
        if os.path.exists(x):
            info("PI - (A)")
            return [Image.open(x)]
        else:
            info("PI - (B)")
            raise ValueError("Cannot process string that is not Image path")

    def handle_numpy(self, obj):
        if obj.dtype == np.float32 or obj.dtype == np.float64:
            info("PI - (C)")
            obj *= 122.5; obj += 122.5
        info("PI - (C2)")
        obj = obj.astype(np.uint8)
        return [Image.fromarray(obj)]

    def __call__(self, x):
        if isinstance(x, str):
            proc_fn = self.handle_string
        elif isinstance(x, np.ndarray):
            proc_fn = self.handle_numpy
        elif isinstance(x, Image.Image):
            info("PI - (D)")
            proc_fn = lambda x: [x]
        else:
            info("PI - (E)")
            raise ValueError(f"Cannot process item of dtype: {type(x)}")
        return proc_fn(x)


class Model:
    """Nbox.Model class designed for inference"""
    def __init__(
        self,
        model: torch.nn.Module,
        category,
        nl_task: str = None,
        tokenizer = None
    ):
        self.model = model
        self.category = category
        self.image_parser = ImageParser()

        if self.category not in ["image"]:
            raise ValueError(f"Category: {self.category} is not supported yet. Raise a PR!")

        # if dtype == "transformers":
        #     assert nl_task is not None, "This is a transformer model, need to provide a task"
        #     assert tokenizer is not None, "This is a transformer model, need to provide a tokenizer"
        #     self.nl_task = nl_task # task when using the transformer model
        #     self.tokenizer = tokenizer # tokenizer for this model

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
            if isinstance(input_object, (list, tuple)):
                _t = []
                for item in input_object:
                    pil_img = self.image_parser(item)[0]
                    _t.append(processing.totensor(pil_img))
                input_tensor = torch.cat(_t, axis = 0)
            else:
                pil_img = self.image_parser(input_object)[0]
                input_tensor = processing.totensor(pil_img)

        out =  self.model(input_tensor)
        return out

    def deploy(self, nbx_api_key):
        # this is a part of one-click to NBX
        raise NotImplementedError()

    def export(self):
        # creates a FastAPI / Flask folder with all the things required to serve this model
        raise NotImplementedError()


# class TextModels(Model):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
# 
#     def pre(self, x):
#         out = self.tokenizer(x, return_tensors = "pt")
# 
#     def generation_task(self, x, **kwargs):
#         pass
# 
#     def task(self, x, **kwargs):
#         if self.task == "generation":
#             out = self.generation_task(x, **kwargs)
#         elif self.task == "masked_lm":
#             out = self.masked_task(x)
#         return
# 
#     def post(self, x):
#         pass
# 
#     def __call__(self, x):
#         pass
# 
# 
# class ImageModels(Model):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
# 
