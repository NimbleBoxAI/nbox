# Plugging NBXApi as another alternative for nbox.Model
# Seemlessly remove boundaries between local and cloud inference

import requests

from nbox.model import ImageParser, TextParser

# main class that calls the NBX Server Models
class NBXApi:
    def __init__(self, model_key: str, nbx_api_key: str):
        """NBXApi would call the NBX Chill Inference API

        Args:
            model_key (str): model (str, optional): key for which to load the model, the structure looks as follows:
                ```
                source/(source/key)::<pre::task::post>
                ```
            nbx_api_key (str): Your Nimblebox API key

        Raises:
            NotImplementedError
        """
        self.model_key = model_key
        self.nbx_api_key = nbx_api_key

        # define the incoming parsers
        self.image_parser = ImageParser()
        self.text_parser = TextParser()
        raise NotImplementedError("WIP")

    def call(self, data):
        r = requests.post(self.url + ":predict", json={"inputs": data})
        return r.json(), r.headers

    def __call__(self, input_object):
        """Just like nbox.Model this can consume any input object

        The entire purpose of this package is to make inference chill.

        Args:
            input_object (Any): input to be processed

        Returns:
            Any: Currently this is output from the API hit
        """
        if self.category == "image":
            # perform parsing for images
            if isinstance(input_object, (list, tuple)):
                data = [self.image_parser(item)[0] for item in input_object]
            else:
                data = [self.image_parser(input_object)[0]]

        elif self.category == "text":
            # perform parsing for text and pass to the model
            data = self.text_parser(input_object, self.tokenizer)

        out = self.call(data)
        return out
