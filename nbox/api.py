# Plugging NBXApi as another alternative for nbox.Model
# Seemlessly remove boundaries between local and cloud inference

import os
import json
import requests
from time import time

import torch
from nbox import processing
from nbox.model import ImageParser, TextParser
from nbox.utils import Console

from pprint import pprint as peepee

URL = os.getenv("NBX_OCD_URL")

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
        
        self.console = Console()

        if self.is_url:
            ovms_meta, nbox_meta = self.prepare_as_url(verbose)

        # define the incoming parsers
        self.image_parser = ImageParser()
        self.text_parser = TextParser(None)

        if self.category == "text":
            import transformers
            model_key = nbox_meta["model_key"].split("::")[0].split("transformers/")[-1]
            tokenizer = transformers.AutoTokenizer.from_pretrained(model_key)
            self.text_parser = TextParser(tokenizer)

    def __repr__(self):
        return f"<nbox.Model: {self.model_key_or_url} >"

    def prepare_as_url(self, verbose):
        self.console.start("Getting model metadata")
        # remove trailing '/'
        r = requests.get(
            f"{URL}/api/model/get_model_meta",
            params = {
                "url": self.model_key_or_url,
                "key": self.nbx_api_key
            }
        )
        try:
            r.raise_for_status()
            content = r.json()["meta"]
            ovms_meta = content["ovms_meta"]
            nbox_meta = json.loads(content["nbox_meta"])
            headers = r.headers
            self.model_key_or_url = self.model_key_or_url[:-1] if self.model_key_or_url.endswith("/") else self.model_key_or_url
        except:
            raise ValueError(f"Could not fetch metadata, please check status: {r.status_code}")


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

    def call(self, data):
        self.console.start("Hitting API")
        if self.verbose:
            print(":0-----------------0:")
            print({k: v.size() for k, v in data.items()})
        data = {k: v.tolist() for k, v in data.items()}
        st = time()
        r = requests.post(self.model_key_or_url + ":predict", json={"inputs": data}, headers={"NBX-KEY": self.nbx_api_key})
        et = time() - st
        try:
            out = r.json()
        except:
            print(r.content)

        self.console.stop(f"Took {et:.3f} seconds!")
        # structure this and 
        return out

    def _handle_input_object(self, input_object):
        """First level handling to convert the input object to a fixed object"""
        if self.category == "image":
            # perform parsing for images
            if isinstance(input_object, (list, tuple)):
                _t = []
                for item in input_object:
                    pil_img = self.image_parser(item)[0]
                    for k, s in self.templates.items():
                        pil_img = pil_img.resize(s[-2:])
                    _t.append(processing.totensor(pil_img))
                input_tensor = torch.cat(_t, axis=0)
            else:
                pil_img = self.image_parser(input_object)[0]
                for k, s in self.templates.items():
                    pil_img = pil_img.resize(s[-2:])
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
