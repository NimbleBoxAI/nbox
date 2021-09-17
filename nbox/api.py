# Plugging CloudModel is cloud alternative for nbox.Model
# Seemlessly remove boundaries between local and cloud inference

import json
import requests
import numpy as np
from time import time

from nbox.utils import Console
from nbox.parsers import ImageParser, TextParser
from nbox.network import URL
from nbox.user import secret

from pprint import pprint as pp


# main class that calls the NBX Server Models
class CloudModel:
    def __init__(self, model_url: str, nbx_api_key: str, verbose: bool = False):
        """CloudModel would call the NBX Chill Inference API

        Args:
            model_url (str): URL from OCD
            nbx_api_key (str): NBX API Key
            verbose (bool, optional): verbose mode. Defaults to False.

        Raises:
            ValueError: if category is not "image" or "text"
        """
        self.nbx_api_key = nbx_api_key
        self.verbose = verbose
        self.console = Console()  # define a console so pretty printing and loading is simple

        assert model_url.startswith("http"), "Are you sure this is a valid URL?"

        # ----------- hit and get metadata for this deployed model
        self.console.start("Getting model metadata")
        r = requests.get(f"{URL}/api/model/get_model_meta", params=f"url={model_url}&key={self.nbx_api_key}")
        try:
            r.raise_for_status()
        except Exception as e:
            print(e)
            raise ValueError(f"Could not fetch metadata, please check status: {r.status_code}")

        content = json.loads(r.content)["meta"]
        ovms_meta = content["ovms_meta"]
        nbox_meta = json.loads(content["nbox_meta"])
        headers = r.headers
        self.category = nbox_meta["spec"]["category"]

        self.model_url = model_url.rstrip("/")  # remove trailing slash

        all_inputs = ovms_meta["metadata"]["signature_def"]["signatureDef"]["serving_default"]["inputs"]
        self.templates = {}
        for node, meta in all_inputs.items():
            self.templates[node] = [int(x["size"]) for x in meta["tensorShape"]["dim"]]

        if verbose:
            print("--------------")
            pp(headers)
            print("--------------")
            pp(ovms_meta)
            print("--------------")
            pp(nbox_meta)
            print("--------------")
            pp(self.templates)
            print("--------------")

        self.console.stop("Cloud infer metadata obtained")
        # ----------- obtained the metadata, now we can create the parser

        # add to secret, if present, this ignores it
        secret.add_ocd(None, self.model_url, nbox_meta, self.nbx_api_key)

        # if category is "text" or if it is dict then any key is "text"
        text_parser = None
        if self.category == "text" or (isinstance(self.category, dict) and any([x == "text" for x in self.category.values()])):
            import transformers

            model_key = nbox_meta["spec"]["model_key"].split("::")[0].split("transformers/")[-1]
            tokenizer = transformers.AutoTokenizer.from_pretrained(model_key)
            max_len = self.templates["input_ids"][-1]
            text_parser = TextParser(tokenizer=tokenizer, max_len=max_len, post_proc_fn=lambda x: x.tolist())

        # define the incoming parsers
        self.image_parser = ImageParser(cloud_infer=True, post_proc_fn=lambda x: x.tolist(), templates=self.templates)
        self.text_parser = text_parser if text_parser else TextParser(None, post_proc_fn=lambda x: x.tolist())

    def __repr__(self):
        return f"<nbox.Model: {self.model_url} >"

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
                        input_dict[k] = self.text_parser(v)
                    else:
                        raise ValueError(f"Unsupported category: {self.category[k]}")
            if verbose:
                for k, v in input_dict.items():
                    print(k, v.shape)

            return input_dict

        if self.category == "image":
            input_obj = self.image_parser(input_object)
            return input_obj

        elif self.category == "text":
            # perform parsing for text and pass to the model
            input_dict = self.text_parser(input_object)
            return input_dict

    def call(self, data):
        self.console.start("Hitting API")
        st = time()
        r = requests.post(self.model_url + ":predict", json={"inputs": data}, headers={"NBX-KEY": self.nbx_api_key})
        et = time() - st

        try:
            r.raise_for_status()
            data_size = len(r.content)
            secret.update_ocd(self.model_url, data_size, len(r.request.body if r.request.body else []))
            out = r.json()

            # first try outputs is a key and we can just get the structure from the list
            if isinstance(out["outputs"], dict):
                out = {k: np.array(v) for k, v in r.json()["outputs"].items()}
            elif isinstance(out["outputs"], list):
                out = np.array(out["outputs"])
            else:
                raise ValueError(f"Outputs must be a dict or list, got {type(out['outputs'])}")
        except:
            out = r.json()
            data_size = 0
            print("Error: ", out)

        self.console.stop(f"Took {et:.3f} seconds!")
        return out

    def __call__(self, input_object, verbose=True):
        """Just like nbox.Model this can consume any input object

        The entire purpose of this package is to make inference chill.

        Args:
            input_object (Any): input to be processed

        Returns:
            Any: Currently this is output from the API hit
        """
        data = self._handle_input_object(input_object=input_object, verbose=verbose)
        out = self.call(data)
        return out
