import torch 
from PIL import Image

from nbox import processing
from nbox import utils

class Model:
    """Nbox.Model class designed for inference"""
    def __init__(
        self,
        model: torch.nn.Module,
        dtype: str = None,
        nl_task: str = None,
        tokenizer = None
    ):
        self.model = model
        self.dtype = dtype

        if dtype == "transformers":
            assert nl_task is not None, "This is a transformer model, need to provide a task"
            assert tokenizer is not None, "This is a transformer model, need to provide a tokenizer"
            self.nl_task = nl_task # task when using the transformer model
            self.tokenizer = tokenizer # tokenizer for this model

    def data_type(self):
        return self.dtype

    def available_dtypes(self):
        """
        Returns a dict of available data types as keys and the respective
        processing method as value.
        """
        return {"image": processing.image_processing}

    def get_model(self):
        return self.model

    def __call__(self, input_object):
        """This is the most important part of this codebase. The `input_object` can be anything from
        a tensor, an image file, filepath as string, string to process as NLP model. This `__call__`
        should understand the different usecases and manage accordingly.
        
        The entire purpose of this package is to make inference chill."""

        if isinstance(input_object, str):
            if self.dtype == "image":
                input_object = utils.get_image(input_object)

        if isinstance(input_object, Image.Image):
            # this is PIL.Image type object
            input_tensor = processing.totensor(input_object)

        out =  self.model(input_tensor)
        return out

    def deploy(self, nbx_api_key):
        # this is a part of one-click to NBX
        raise NotImplementedError()

    def export(self):
        # creates a FastAPI / Flask folder with all the things required to serve this model
        raise NotImplementedError()


class TextModels(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def pre(self, x):
        out = self.tokenizer(x, return_tensors = "pt")

    def generation_task(self, x, **kwargs):
        pass

    def task(self, x, **kwargs):
        if self.task == "generation":
            out = self.generation_task(x, **kwargs)
        elif self.task == "masked_lm":
            out = self.masked_task(x)
        return

    def post(self, x):
        pass

    def __call__(self, x):
        pass


class ImageModels(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
