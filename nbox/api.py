# Plugging NBXApi as another alternative for nbox.Model
# Seemlessly remove boundaries between local and cloud inference

import requests
from time import time

import torch
from nbox import processing
from nbox.model import ImageParser, TextParser

from pprint import pprint as peepee

# main class that calls the NBX Server Models
class NBXApi:
    def __init__(self, model_key_or_url: str, nbx_api_key: str, category: str, verbose: bool = False):
        """NBXApi would call the NBX Chill Inference API

        Args:
            model_key_or_url (str): URL from OCD or model_key
            nbx_api_key (str): NBX API Key
            category (str): "image" or "text"
            verbose (bool, optional): verbose mode. Defaults to False.

        Raises:
            ValueError: if category is not "image" or "text"
        """
        self.nbx_api_key = nbx_api_key
        self.category = category
        self.verbose = verbose

        self.is_url = model_key_or_url.startswith("https") or model_key_or_url.startswith("http")
        self.is_on_nbx = "api.nimblebox.ai" in model_key_or_url

        assert self.is_url, "Currently only URLs are accepted"
        assert self.nbx_api_key, "Invalid `nbx_api_key` found"

        if self.is_url:
            # remove trailing '/'
            model_key_or_url = model_key_or_url[:-1] if model_key_or_url.endswith("/") else model_key_or_url
            self.model_key_or_url = model_key_or_url
            print(self.model_key_or_url + "/metadata")
            r = requests.get(url=self.model_key_or_url + "/metadata", headers={"NBX-KEY": self.nbx_api_key})
            try:
                r.raise_for_status()
            except:
                raise ValueError(f"Could not fetch metadata, please check: {r.content}")

            content = r.json()
            headers = r.headers

            if verbose:
                peepee(content)
                peepee(headers)

            all_inputs = content["metadata"]["signature_def"]["signatureDef"]["serving_default"]["inputs"]
            self.templates = {}
            for node, meta in all_inputs.items():
                self.templates[node] = [int(x["size"]) for x in meta["tensorShape"]["dim"]]

            if verbose:
                peepee(self.templates)

        # define the incoming parsers
        self.image_parser = ImageParser()
        self.text_parser = TextParser(None)

    def __repr__(self):
        return f"<nbox.Model: {self.model_key_or_url} >"

    def call(self, data):
        if self.verbose:
            print(":0-----------------0:")
            print({k: v.size() for k, v in data.items()})
        data = {k: v.tolist() for k, v in data.items()}
        st = time()
        r = requests.post(self.model_key_or_url + ":predict", json={"inputs": data}, headers={"NBX-KEY": self.nbx_api_key})
        et = time() - st
        return r.json(), r.headers, et

    def _handle_input_object(self, input_object):
        """First level handling to convert the input object to a fixed object"""
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
                pil_img = pil_img.resize((224, 224))
                input_tensor = processing.totensor(pil_img)

            data = {}
            for k in self.templates:
                data[k] = input_tensor
            return data

        elif self.category == "text":
            # perform parsing for text and pass to the model
            input_dict = self.text_parser(input_object, self.tokenizer)
            input_dict = {k: v for k, v in input_dict.items()}
            return input_dict

    def __call__(self, input_object):
        """Just like nbox.Model this can consume any input object

        The entire purpose of this package is to make inference chill.

        Args:
            input_object (Any): input to be processed

        Returns:
            Any: Currently this is output from the API hit
        """
        data = self._handle_input_object(input_object=input_object)
        out = self.call(data)
        return out
