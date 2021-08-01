import torch
from PIL import Image
import numpy as np

from aibox import utils

class Processing:
    # Define processing methods for different input types, eg video, images,
    # text, etc... so they can be called later in the Models class in the
    # __call__ method.

    # Probably also add something like transformation from torchvision.transforms
    @staticmethod
    def image_processing(input_path):
        img = Image.open(input_path).convert("RGB")
        img = torch.tensor(np.asarray(img))
        img = img.permute(2, 0, 1).unsqueeze(0)
        return img.float()

    @staticmethod
    def totensor(image):
        img = torch.tensor(np.asarray(image))
        img = img.permute(2, 0, 1).unsqueeze(0)
        return img.float()


class Model:
    def __init__(self, model: torch.nn.Module, dtype: str = None):
        self.model = model
        self.dtype = dtype

    def data_type(self):
        return self.dtype

    def available_dtypes(self):
        """
        Returns a dict of available data types as keys and the respective
        processing method as value.
        """
        return {"image": Processing.image_processing}

    def get_model(self):
        return self.model

    def __call__(self, input_object):
        """This is the most important part of this codebase. The `input_object` can be anything from
        a tensor, an image file, filepath as string, string to process as NLP model. This `__call__`
        should understand the different usecases and manage accordingly.
        
        The entire purpose of this package is to make inference chill."""

        if isinstance(input_object, Image.Image):
            # this is PIL.Image type object
            img = Processing.totensor(input_object)
        elif isinstance(input_object, str):
            # this is a string
            # for now the assumption is that it is an image to be loaded
            img = utils.get_image(input_object)
            img = Processing.totensor(img)

        input_tensor = img

        return self.model(input_tensor)

    def deploy(self, nbx_api_key):
        # this is a part of one-click to NBX
        raise NotImplementedError()

    def export(self):
        # creates a FastAPI / Flask folder with all the things required to serve this model
        raise NotImplementedError()

