# this file has the code for nbox.Model that is the holy grail of the project
# seemlessly remove boundaries between local and cloud inference
# from nbox==0.1.10 nbox.Model handles both local and remote models

import json
import requests
from time import time
from typing import Any
from pprint import pprint as pp

import torch
import numpy as np

from nbox import network
from nbox.utils import Console
from nbox.parsers import ImageParser, TextParser
from nbox.network import URL
from nbox.user import secret
from nbox.framework.pytorch import get_meta


class Model:
    def __init__(
        self,
        model_or_model_url,
        nbx_api_key: str = None,
        category: str = None,
        tokenizer=None,
        model_key: str = None,
        model_meta: dict = None,
        verbose: bool = False,
    ):
        """Nbox.Model class designed for inference

        Args:
            model (torch.nn.Module): Model to be wrapped.
            category (str): Catogory of the model task.
            tokenizer (str, optional): Tokenizer model if this is an NLP category. Defaults to None.
            model_key (str, optional): key used to initialise this model. Defaults to None.

        Raises:
            ValueError: If the category is incorrect
            AssertionError: When items required for the each category are not available
        """
        # for now just add everything, we will add more things later
        self.model_or_model_url = model_or_model_url
        self.nbx_api_key = nbx_api_key
        self.category = category
        self.tokenizer = tokenizer
        self.model_key = model_key
        self.model_meta = model_meta
        self.verbose = verbose

        # define the console, either it get's used or lays unused, doesn't matter
        self.console = Console()

        nbox_meta = None
        if isinstance(model_or_model_url, str):
            self.__on_cloud = True
            assert isinstance(nbx_api_key, str), "Nbx API key must be a string"
            assert nbx_api_key.startswith("nbxdeploy_"), "Not a valid NBX Api key, please check again."
            assert model_or_model_url.startswith("http"), "Are you sure this is a valid URL?"

            self.model_url = model_or_model_url.rstrip("/")

            # when on_cloud, there is no need to load tokenizers, categories, and model_meta
            # this all gets fetched from the deployment node
            nbox_meta, category = self.fetch_meta_from_nbx_cloud()
            self.category = category

            # if category is "text" or if it is dict then any key is "text"
            tokenizer = None
            max_len = None
            if self.category == "text" or (isinstance(self.category, dict) and any([x == "text" for x in self.category.values()])):
                import transformers

                model_key = nbox_meta["spec"]["model_key"].split("::")[0].split("transformers/")[-1]
                tokenizer = transformers.AutoTokenizer.from_pretrained(model_key)
                max_len = self.templates["input_ids"][-1]

            self.image_parser = ImageParser(cloud_infer=True, post_proc_fn=lambda x: x.tolist(), templates=self.templates)
            self.text_parser = TextParser(tokenizer=tokenizer, max_len=max_len, post_proc_fn=lambda x: x.tolist())

        else:
            self.__on_cloud = False
            assert isinstance(model_or_model_url, torch.nn.Module), "model_or_model_url must be a torch.nn.Module "

            self.model = model_or_model_url
            self.category = category
            self.model_key = model_key
            self.model_meta = model_meta  # this is a big dictionary (~ same) as TF-Serving metadata

            assert self.category is not None, "Category must be provided"

            # initialise all the parsers
            self.image_parser = ImageParser(post_proc_fn=lambda x: torch.from_numpy(x).float())
            self.text_parser = TextParser(tokenizer=tokenizer, post_proc_fn=lambda x: torch.from_numpy(x).int())

            if isinstance(self.category, dict):
                assert all([v in ["image", "text"] for v in self.category.values()])
            else:
                if self.category not in ["image", "text"]:
                    raise ValueError(f"Category: {self.category} is not supported yet. Raise a PR!")

            if self.category == "text":
                assert tokenizer != None, "tokenizer cannot be none for a text model!"

        self.nbox_meta = nbox_meta

    def fetch_meta_from_nbx_cloud(self):
        self.console.start("Getting model metadata")
        r = requests.get(f"{URL}/api/model/get_model_meta", params=f"url={self.model_or_model_url}&key={self.nbx_api_key}")
        try:
            r.raise_for_status()
        except Exception as e:
            print(e)
            raise ValueError(f"Could not fetch metadata, please check status: {r.status_code}")

        # start getting the metadata, note that we have completely dropped using OVMS meta and instead use nbox_meta
        content = json.loads(r.content)["meta"]
        nbox_meta = json.loads(content["nbox_meta"])
        category = nbox_meta["spec"]["category"]

        all_inputs = nbox_meta["metadata"]["inputs"]
        self.templates = {}
        for node, meta in all_inputs.items():
            self.templates[node] = [int(x["size"]) for x in meta["tensorShape"]["dim"]]

        if self.verbose:
            print("--------------")
            pp(nbox_meta)
            print("--------------")
            pp(self.templates)
            print("--------------")

        self.console.stop("Cloud infer metadata obtained")

        # add to secret, if present, this ignores it
        secret.add_ocd(None, self.model_url, nbox_meta, self.nbx_api_key)

        return nbox_meta, category

    def eval(self):
        assert not self.__on_cloud
        self.model.eval()

    def train(self):
        assert not self.__on_cloud
        self.model.train()

    def __repr__(self):
        if not self.__on_cloud:
            return f"<nbox.Model: {repr(self.model)} >"
        else:
            return f"<nbox.Model: {self.model_url} >"

    def _handle_input_object(self, input_object):
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
            return input_dict

        elif self.category == "image":
            input_obj = self.image_parser(input_object)
            return input_obj

        elif self.category == "text":
            # perform parsing for text and pass to the model
            input_dict = self.text_parser(input_object)
            return input_dict

    def __call__(self, input_object: Any, return_inputs=False):
        """This is the most important part of this codebase. The `input_object` can be anything from
        a tensor, an image file, filepath as string, string to process as NLP model. This `__call__`
        should understand the different usecases and manage accordingly.

        The current idea is that what ever the input, based on the category (image, text, audio, smell)
        it will be parsed through dedicated parsers that can make ingest anything.

        The entire purpose of this package is to make inference chill.

        Args:
            input_object (Any): input to be processed
            return_inputs (bool, optional): whether to return the inputs or not. Defaults to False.
            verbose (bool, optional): whether to print the inputs or not. Defaults to False.

        Returns:
            Any: currently this is output from the model, so if it is tensors and return dicts.
        """

        model_input = self._handle_input_object(input_object=input_object)

        if self.__on_cloud:
            self.console.start("Hitting API")
            st = time()
            r = requests.post(self.model_url + ":predict", json={"inputs": model_input}, headers={"NBX-KEY": self.nbx_api_key})
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

        else:
            with torch.no_grad():
                if isinstance(model_input, dict):
                    out = self.model(**model_input)
                else:
                    assert isinstance(model_input, torch.Tensor)
                    out = self.model(model_input)

            if self.model_meta is not None and self.model_meta.get("metadata", False) and self.model_meta["metadata"].get("outputs", False):
                outputs = self.model_meta["metadata"]["outputs"]
                if not isinstance(out, torch.Tensor):
                    assert len(outputs) == len(out)
                    out = {k: v.numpy() for k, v in zip(outputs, out)}
                else:
                    out = {k: v.numpy() for k, v in zip(outputs, [out])}

        if return_inputs:
            return out, model_input
        return out

    def get_nbox_meta(self, input_object):
        # this function gets the nbox metadata for the the current model, based on the input_object
        assert not self.__on_cloud, "This function is not supported when using cloud infer"

        self.eval()  # covert to eval mode
        model_output, model_input = self(input_object, return_inputs=True)

        # need to convert inputs and outputs to list / tuple
        dynamic_axes_dict = {
            0: "batch_size",
        }
        if self.category == "text":
            dynamic_axes_dict[1] = "sequence_length"

        # need to convert inputs and outputs to list / tuple
        if isinstance(model_input, dict):
            args = tuple(model_input.values())
            input_names = tuple(model_input.keys())
            input_shapes = tuple([tuple(v.shape) for k, v in model_input.items()])
        elif isinstance(model_input, torch.Tensor):
            args = tuple([model_input])
            input_names = tuple(["input_0"])
            input_shapes = tuple([tuple(model_input.shape)])
        dynamic_axes = {i: dynamic_axes_dict for i in input_names}

        if isinstance(model_output, dict):
            output_names = tuple(model_output.keys())
            output_shapes = tuple([tuple(v.shape) for k, v in model_output.items()])
            model_output = tuple(model_output.values())
        elif isinstance(model_output, (list, tuple)):
            mo = model_output[0]
            if isinstance(mo, dict):
                # cases like [{"output_0": tensor, "output_1": tensor}]
                output_names = tuple(mo.keys())
                output_shapes = tuple([tuple(v.shape) for k, v in mo.items()])
            else:
                output_names = tuple([f"output_{i}" for i, x in enumerate(model_output)])
                output_shapes = tuple([tuple(v.shape) for v in model_output])
        elif isinstance(model_output, torch.Tensor):
            output_names = tuple(["output_0"])
            output_shapes = (tuple(model_output.shape),)

        meta = get_meta(input_names, input_shapes, args, output_names, output_shapes, model_output)
        out = {
            "input_names": input_names,
            "input_shapes": input_shapes,
            "args": args,
            "output_names": output_names,
            "output_shapes": output_shapes,
            "outputs": model_output,
            "dynamic_axes": dynamic_axes,
        }
        return meta, out

    def deploy(self, input_object: Any, model_name: str = None, cache_dir: str = None):
        """OCD your model on NBX platform.

        Args:
            input_object (Any): input to be processed
            model_name (str, optional): custom model name for this model. Defaults to None.
            cache_dir (str, optional): Custom caching directory. Defaults to None.
        """
        # user will always have to pass the input_object
        nbox_meta, meta_dict = self.get_nbox_meta(input_object)

        # OCD baby!
        network.one_click_deploy(
            model_key=self.model_key,
            model=self.model,
            category=self.category,
            model_name=model_name,
            cache_dir=cache_dir,
            **meta_dict,
        )
