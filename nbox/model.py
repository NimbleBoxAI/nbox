# this file has the code for nbox.Model that is the holy grail of the project

import os
import json
import requests
from time import time
from typing import Any, Union
from pprint import pprint as pp

import torch
import numpy as np

from nbox import utils
from nbox.utils import Console
from nbox.parsers import ImageParser, TextParser
from nbox.network import one_click_deploy
from nbox.user import secret
from nbox.framework import get_meta
from nbox.framework import pytorch as frm_pytorch, sklearn as frm_skl


class Model:
    def __init__(
        self,
        model_or_model_url: Any,
        nbx_api_key: str = None,
        category: str = None,
        tokenizer=None,
        model_key: str = None,
        model_meta: dict = None,
        verbose: bool = False,
    ):
        """Model class designed for inference. Seemlessly remove boundaries between local and cloud inference
        from ``nbox==0.1.10`` ``nbox.Model`` handles both local and remote models

        Usage:

            .. code-block:: python

                from nbox import Model

                # when on NBX-Deploy
                model = Model("https://nbx.cloud/model/url", "nbx_api_key")

                # when loading a scikit learn model
                from sklearn.datasets import load_iris
                from sklearn.ensemble import RandomForestClassifier
                iris = load_iris()
                clr = RandomForestClassifier()
                clr.fit(iris.data, iris.target)
                model = nbox.Model(clr)

                # when loading a pytorch model
                import torch

                class DoubleInSingleOut(torch.nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.f1 = torch.nn.Linear(2, 4)
                        self.f2 = torch.nn.Linear(2, 4)
                        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

                    def forward(self, x, y):
                        out = self.f1(x) + self.f2(y)
                        logit_scale = self.logit_scale.exp()
                        out = logit_scale - out @ out.t()
                        return out

                model = nbox.Model(
                    DoubleInSingleOut(),
                    category = {"x": "image", "y": "image"} # <- this is pre-proc type for input
                )
                # this is not the best approach, but it works for now


        Args:
            model_or_model_url (Any): Model to be wrapped or model url
            nbx_api_key (str, optional): API key for this deployed model
            category (str, optional): Input categories for each input type to the model
            tokenizer ([transformers.PreTrainedTokenizer], optional): If this is a text model then tokenizer for this. Defaults to None.
            model_key (str, optional): With what key is this model initialised, useful for public models. Defaults to None.
            model_meta (dict, optional): Extra metadata when starting the model. Defaults to None.
            verbose (bool, optional): If true provides detailed prints. Defaults to False.

        Raises:
            ValueError: If any category is "text" and tokenizer is not provided.
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
            self.__framework = "nbx"

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

        # <class 'sklearn.ensemble._forest.RandomForestClassifier'>
        elif "sklearn" in str(type(model_or_model_url)):
            self.__framework = "sk"

        else:
            assert isinstance(model_or_model_url, torch.nn.Module), "model_or_model_url must be a torch.nn.Module "
            self.__framework = "pt"
            assert self.category is not None, "Category for inputs must be provided, when loading model manually"

            self.category = category
            self.model_key = model_key
            self.model_meta = model_meta  # this is a big dictionary (~ same) as TF-Serving metadata

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
        self.__device = "cpu"
        self.cache_dir = None

    def fetch_meta_from_nbx_cloud(self):
        """When this is on NBX-Deploy cloud, fetch the metadata from the deployment node."""
        self.console.start("Getting model metadata")
        URL = secret.get("nbx_url")
        r = requests.get(f"{URL}/api/model/get_model_meta", params=f"url={self.model_or_model_url}&key={self.nbx_api_key}")
        try:
            r.raise_for_status()
        except Exception as e:
            self.console.stop(e)
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
        """if underlying model has eval method, call it"""
        if hasattr(self.model_or_model_url, "eval"):
            self.model_or_model_url.eval()

    def train(self):
        """if underlying model has train method, call it"""
        if hasattr(self.model_or_model_url, "train"):
            self.model_or_model_url.train()

    def __repr__(self):
        return f"<nbox.Model: {self.model_or_model_url} >"

    def _handle_input_object(self, input_object):
        """First level handling to convert the input object to a fixed object"""
        # in case of scikit learn user must ensure that the input_object is model_input
        if self.__framework == "sk":
            return input_object

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

        elif self.category == None and isinstance(input_object, np.ndarray):
            # this has to be handled better -> to be fixed till 0.2.1
            return {"input_0": input_object.tolist()}

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

        Returns:
            Any: currently this is output from the model, so if it is tensors and return dicts.
        """

        model_input = self._handle_input_object(input_object=input_object)

        if self.__framework == "nbx":
            self.console.start("Hitting API")
            st = time()
            # OVMS has :predict endpoint and nbox has /predict
            _p = "/" if "export_type" in self.nbox_meta["spec"] else ":"
            r = requests.post(self.model_url + f"{_p}predict", json={"inputs": model_input}, headers={"NBX-KEY": self.nbx_api_key})
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
                print(r.content)
                out = r.json()
                data_size = 0
                print("Error: ", out)

            self.console.stop(f"Took {et:.3f} seconds!")

        elif self.__framework == "sk":
            if str(type(self.model_or_model_url)).startswith("sklearn.neighbors"):
                out = self.model_or_model_url.kneighbors
            out = self.model_or_model_url.predict(model_input)

        elif self.__framework == "pt":
            with torch.no_grad():
                if isinstance(model_input, dict):
                    model_input = {k: v.to(self.__device) for k, v in model_input.items()}
                    out = self.model_or_model_url(**model_input)
                else:
                    assert isinstance(model_input, torch.Tensor)
                    model_input = model_input.to(self.__device)
                    out = self.model_or_model_url(model_input)

                # # bring back to cpu
                # if isinstance(out, (tuple, list)):
                #     out = [x.to("cpu") for x in out]
                # elif isinstance(out, torch.Tensor):
                #     out = out.to("cpu")
                # elif isinstance(out, dict):
                #     out = {k: v.to("cpu") for k, v in out.items()}

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
        """Get the nbox meta and trace args for the model with the given input object

        Args:
            input_object (Any): input to be processed
        """
        # this function gets the nbox metadata for the the current model, based on the input_object
        if self.__framework == "nbx":
            return self.nbox_meta

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
            model_inputs = tuple(model_input.values())
            input_names = tuple(model_input.keys())
            input_shapes = tuple([tuple(v.shape) for k, v in model_input.items()])
        elif isinstance(model_input, (torch.Tensor, np.ndarray)):
            model_inputs = tuple([model_input])
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
        elif isinstance(model_output, (torch.Tensor, np.ndarray)):
            output_names = tuple(["output_0"])
            output_shapes = (tuple(model_output.shape),)

        meta = get_meta(input_names, input_shapes, model_inputs, output_names, output_shapes, model_output)
        out = {
            "args": model_inputs,
            "outputs": model_output,
            "input_shapes": input_shapes,
            "output_shapes": output_shapes,
            "input_names": input_names,
            "output_names": output_names,
            "dynamic_axes": dynamic_axes,
        }
        return meta, out

    def export(self, input_object: Any, export_type: str = "onnx", model_name: str = None, cache_dir=None, return_convert_args=False):
        """Export the model to a particular kind of DAG (ie. like onnx, torchscript, etc.)

        Raises appropriate assertion errors for strict checking of inputs

        Args:
            input_object (Any): input to be processed
            export_type (str, optional): export type, see errors (use this first)
            model_name (str, optional): will be saved to cache_dir/model_name.<export_type>
            cache_dir (str, optional): cache dir where to dump this file
            return_convert_args (bool, optional): if True, this structured input to the model

        Usage:

            .. code-block:: python

                export_model_path, model_name, nbox_meta = model.export(some_input, "onnx", "my_model_name")

        Returns:
            tuple: ``export_model_path, model_name, nbox_meta`` and ``convert_args`` if ``return_convert_args==True``
        """
        # First Step: check the args and see if conditionals are correct or not
        def __check_conditionals():
            assert self.__framework != "nbx", "This model is already deployed on the cloud"
            assert export_type in ["onnx", "torchscript"], "Export type must be onnx or torchscript"
            if self.__framework == "sk":
                assert export_type == "onnx", "Only ONNX export is supported for scikit-learn Framework"

        # perform sanity checks on the input values
        __check_conditionals()

        nbox_meta, export_kwargs = self.get_nbox_meta(input_object)
        _m_hash = utils.hash_(self.model_key)
        model_name = model_name if model_name is not None else f"{utils.get_random_name()}-{_m_hash[:4]}".replace("-", "_")

        # intialise the console logger
        console = utils.Console()
        console.rule(f"Exporting {model_name}")
        cache_dir = "/tmp" if self.cache_dir is None else self.cache_dir

        # convert the model -> create a the spec, get the actual method for conversion
        console(f"model_name: {model_name}")
        console._log(f"Export type: ", export_type)
        spec = {
            "category": self.category,
            "model_key": self.model_key,
            "name": model_name,
            "src_framework": self.__framework,
            "export_type": export_type,
        }
        nbox_meta = {"metadata": nbox_meta, "spec": spec}
        export_model_path = os.path.abspath(utils.join(cache_dir, _m_hash))

        src_module = frm_skl if self.__framework == "sk" else frm_pytorch
        if export_type == "onnx":
            export_model_path += ".onnx"
            export_fn = src_module.export_to_onnx
        elif export_type == "torchscript":  # can only be deployed using pytorch framework
            export_model_path += ".torchscript"
            export_fn = frm_pytorch.export_to_torchscript

        console.start(f"Converting using: {export_fn}")
        export_fn(model=self.model_or_model_url, export_model_path=export_model_path, **export_kwargs)
        console.stop("Conversion Complete")
        console._log("nbox_meta:", nbox_meta)

        # construct the output
        fn_out = [export_model_path, model_name, nbox_meta]
        if return_convert_args:
            # https://docs.openvinotoolkit.org/latest/openvino_docs_MO_DG_prepare_model_convert_model_Converting_Model_General.html
            input_ = ",".join(export_kwargs["input_names"])
            input_shape = ",".join([str(list(x.shape)).replace(" ", "") for x in export_kwargs["args"]])
            convert_args = f"--data_type=FP32 --input_shape={input_shape} --input={input_} "

            if self.category == "image":
                # mean and scale have to be defined for every single input
                # these values are calcaulted from uint8 -> [-1,1] -> ImageNet scaling -> uint8
                mean_values = ",".join([f"{name}[182,178,172]" for name in export_kwargs["input_names"]])
                scale_values = ",".join([f"{name}[28,27,27]" for name in export_kwargs["input_names"]])
                convert_args += f"--mean_values={mean_values} --scale_values={scale_values}"

            console._log(convert_args)
            fn_out = fn_out + [convert_args]

        return fn_out

    def deploy(
        self,
        input_object: Any,
        model_name: str = None,
        cache_dir: str = None,
        wait_for_deployment: bool = False,
        runtime: str = "onnx",
        deployment_type: str = "ovms2",
    ):
        """NBX-Deploy (read more one: https://docs.nimblebox.ai/nbox/nbox-intro/quickstart-with-deployment)

        This deploys the current model onto our managed K8s clusters. This tight product service integration
        is very crucial for us and is the best way to make deploy a model for usage.

        Raises appropriate assertion errors for strict checking of inputs

        Args:
            input_object (Any): input to be processed
            model_name (str, optional): custom model name for this model
            cache_dir (str, optional): custom caching directory
            wait_for_deployment (bool, optional): wait for deployment to complete
            runtime (str, optional): runtime to use for deployment
            deployment_type (str, optional): deployment type
        """
        # First Step: check the args and see if conditionals are correct or not
        def __check_conditionals():
            assert self.__framework != "nbx", "This model is already deployed on the cloud"
            assert deployment_type in ["ovms2", "nbox"], f"Only OpenVino and Nbox-Serving is supported got: {deployment_type}"
            if self.__framework == "sk":
                assert deployment_type == "nbox", "Only ONNX Runtime is supported for scikit-learn Framework"

        # perform sanity checks on the input values
        __check_conditionals()

        # user will always have to pass the input_object
        export_model_path, model_name, nbox_meta, convert_args = self.export(
            input_object, runtime, model_name, cache_dir, return_convert_args=True
        )

        # OCD baby!
        out = one_click_deploy(
            export_model_path=export_model_path,
            deployment_type=deployment_type,
            nbox_meta=nbox_meta,
            model_name=model_name,
            wait_for_deployment=wait_for_deployment,
            convert_args=convert_args,
        )

        if out != None:
            return out
