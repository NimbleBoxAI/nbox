# Plugging NBXApi as another alternative for nbox.Model
# Seemlessly remove boundaries between local and cloud inference

import os
import json
import requests
import numpy as np
from time import time

import torch

from nbox.utils import Console
from nbox.parsers import ImageParser, TextParser

from pprint import pprint as peepee

URL = os.getenv("NBX_OCD_URL")

from PIL import Image


class WebParser:
    def __init__(self, image_parser, text_parser):
        self.image_parser = image_parser
        self.text_parser = text_parser

    def __repr__(self):
        return "<WebParser: {}, {}>".format(self.image_parser, self.text_parser)

    def format_image(self, x, resize_image=None):
        # take this torch.Tensor object image convert it back to PIL.Image and return numpy array
        assert len(x.shape) == 4, "shape must be [N,C,H,W] got {}".format(x.shape)
        x = x.permute(0, 2, 3, 1)
        x = x.cpu().numpy()
        x *= 122.5
        x += 122.5
        x[x > 255] = 255
        x[x < 0] = 0
        x = x.astype(np.uint8)
        if resize_image is not None:
            img = Image.fromarray(x[0])
            img = img.resize(resize_image)
            x = torch.from_numpy(np.array(img)).to(torch.uint8)
        x = x.reshape(1, *x.shape)
        x = x.permute(0, 3, 1, 2)
        x = x.tolist()
        return x


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

        self.console = Console()

        if self.is_url:
            ovms_meta, nbox_meta = self.prepare_as_url(verbose, model_key_or_url)

        # define the incoming parsers
        self.image_parser = ImageParser(cloud_infer=False)
        self.text_parser = TextParser(tokenizer=None)

        if self.category == "text":
            import transformers

            model_key = nbox_meta["model_key"].split("::")[0].split("transformers/")[-1]
            tokenizer = transformers.AutoTokenizer.from_pretrained(model_key)
            self.text_parser = TextParser(tokenizer)

        self.web_parser = WebParser(self.image_parser, self.text_parser)

    def __repr__(self):
        return f"<nbox.Model: {self.model_key_or_url} >"

    def prepare_as_url(self, verbose, model_key_or_url):
        self.console.start("Getting model metadata")
        r = requests.get(f"{URL}/api/model/get_model_meta", params={"url": model_key_or_url + "/", "key": self.nbx_api_key})
        try:
            r.raise_for_status()
        except Exception as e:
            print(e)
            raise ValueError(f"Could not fetch metadata, please check status: {r.status_code}")

        content = json.loads(r.content)["meta"]
        ovms_meta = content["ovms_meta"]
        nbox_meta = content["nbox_meta"]
        headers = r.headers

        # remove trailing '/'
        self.model_key_or_url = model_key_or_url[:-1] if model_key_or_url.endswith("/") else model_key_or_url

        if verbose:
            peepee(headers)
            print("--------------")
            peepee(ovms_meta)
            print("--------------")
            peepee(nbox_meta)

        all_inputs = ovms_meta["metadata"]["signature_def"]["signatureDef"]["serving_default"]["inputs"]
        self.templates = {}
        for node, meta in all_inputs.items():
            self.templates[node] = [int(x["size"]) for x in meta["tensorShape"]["dim"]]

        if verbose:
            peepee(self.templates)

        self.console.stop("Cloud infer metadata obtained")
        return ovms_meta, nbox_meta

    def call(self, data, verbose=False):
        self.console.start("Hitting API")
        st = time()
        r = requests.post(self.model_key_or_url + ":predict", json={"inputs": data}, headers={"NBX-KEY": self.nbx_api_key})
        et = time() - st

        try:
            r.raise_for_status()
            out = r.json()
        except:
            print(r.content)

        self.console.stop(f"Took {et:.3f} seconds!")
        # structure out and return

        return out

    def _handle_input_object(self, input_object, verbose=False):
        """First level handling to convert the input object to a fixed object"""
        if isinstance(self.category, dict):
            assert isinstance(input_object, dict), "If category is a dict then input must be a dict"
            # check for same keys
            assert set(input_object.keys()) == set(self.category.keys())
            input_dict = {}
            for k, v in input_object.items():
                if k in self.category:
                    if self.category[k] == "image":
                        input_dict[k] = self.image_parser(v)
                    elif self.category[k] == "text":
                        input_dict[k] = self.text_parser(v)["input_ids"]
                    else:
                        raise ValueError(f"Unsupported category: {self.category[k]}")
            if verbose:
                for k, v in input_dict.items():
                    print(k, v.shape)

            input_dict = {k: v.tolist() for k, v in input_dict.items()}
            return input_dict

        if self.category == "image":
            input_obj = self.image_parser(input_object)
            input_obj = self.web_parser.format_image(input_obj, resize_image=[224, 224])
            return input_obj

        elif self.category == "text":
            # perform parsing for text and pass to the model
            input_dict = self.text_parser(input_object)
            input_dict = {k: v.tolist() for k, v in input_dict.items()}
            if verbose:
                for k, v in input_dict.items():
                    print(k, v.shape)
            return input_dict

    def __call__(self, input_object, verbose=True):
        """Just like nbox.Model this can consume any input object

        The entire purpose of this package is to make inference chill.

        Args:
            input_object (Any): input to be processed

        Returns:
            Any: Currently this is output from the API hit
        """
        data = self._handle_input_object(input_object=input_object, verbose=verbose)
        out = self.call(data, verbose=verbose)
        return out
