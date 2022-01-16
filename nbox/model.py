# this file has the code for nbox.Model that is the holy grail of the project

import re
import os
import json
import shutil
import inspect
import tarfile
import requests
import numpy as np
from glob import glob
from time import time
from tempfile import gettempdir
from pprint import pprint as pp

# import nbox things
from . import utils

# required for lazy loading --> the underlying framework as the information
# on what modules are to be loaded using IMPORTS variable defined in each framework
# conditional
from .framework import get_meta, __all__ as framework_exports
for x in framework_exports[1:]:
    exec(f"from .framework import {x} as frm_{x}")
    IMPORTS = eval(f"frm_{x}.IMPORTS")
    for y in IMPORTS:
        exec(f"import {y}")

from .network import deploy_model
from .parsers import ImageParser, TextParser
from .auth import secret

import logging

logger = logging.getLogger()


class GenericMixin:
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


class Model(GenericMixin):
    # @cshubhamrao suggested lazy evaulation of packages that would avoid initialization Errors
    @utils.isthere("torch", "sklearn", "onnxruntime", "skl2onnx")
    def __init__(self, model_or_model_url, nbx_api_key=None, category=None, tokenizer=None, model_key=None, nbox_meta=None, cache_dir = None, verbose=False):
        """Model class designed for inference. Seemlessly remove boundaries between local and cloud inference
        from ``nbox==0.1.10`` ``nbox.Model`` handles both local and remote models. from ``nbox==0.4.0`` can load
        and serve ONNX models.

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

        This class can process the following types of models:

        .. code-block::

            ``torch.nn.Module`` |           |           |
                    ``sklearn`` | __init__  > serialise > S-*
                 ``nbx-deploy`` |___________|___________|____
                     ``S-onnx`` |              |
              ``S-torchscript`` | .deserialise > __init__
                      ``S-pkl`` |______________|___________

        Serialised models are 1-3 models that have been serialised to load later. This is especially useful for
        ``nbox-serving``, one of the server types we use in NBX Deploy, yes production. This is also part of
        YoCo. Since there is a decidate serialise function, we should have one for deserialisation as well. Use
        ``nbox.Model.desirialise`` to load a serialised model.
        """
        self.model_or_model_url = model_or_model_url
        self.nbx_api_key = nbx_api_key
        self.verbose = verbose

        # here we have the main conditionals for loading the models
        # from code perspective avoid "self" assignment made inside each block,
        # but rather at the end of the blocks
        __framework = None
        nbox_meta = nbox_meta
        image_parser = None
        text_parser = None

        # the tricky part about lazy loading is that it needs to be done in the right order
        # so errors are seen by developer only when needed. So first we check if it is
        # a) deployed model or not
        # b) is a sklearn model
        # else: # this has to be run in try catch blocks
        #  c) is a torch model
        #  d) is a onnx model

        if isinstance(model_or_model_url, str) and model_or_model_url.startswith("http"):
            logger.info(f"Trying to load from url: {model_or_model_url}")
            __framework = "nbx"

            assert isinstance(nbx_api_key, str), "Nbx API key must be a string"
            assert nbx_api_key.startswith("nbxdeploy_"), "Not a valid NBX Api key, please check again."
            assert model_or_model_url.startswith("http"), "Are you sure this is a valid URL?"

            self.model_url = model_or_model_url.rstrip("/")
            nbox_meta, templates = self.fetch_meta_from_nbx_cloud()
            category = nbox_meta["spec"]["category"]

            # if category is "text" or if it is dict then any key is "text"
            tokenizer = None
            max_len = None
            if category == "text" or (isinstance(category, dict) and any([x == "text" for x in category.values()])):
                import transformers

                model_key = nbox_meta["spec"]["model_key"].split("::")[0].split("transformers/")[-1]
                tokenizer = transformers.AutoTokenizer.from_pretrained(model_key)
                max_len = templates["input_ids"][-1]

            image_parser = ImageParser(cloud_infer=True, post_proc_fn=lambda x: x.tolist(), templates=templates)
            text_parser = TextParser(tokenizer=tokenizer, max_len=max_len, post_proc_fn=lambda x: x.tolist())

            self.templates = templates

        elif "sklearn" in str(type(model_or_model_url)):
            logger.info(f"Trying to load from sklearn model: {model_or_model_url}")
            __framework = "sklearn"

        else:
            done = False
            try:
                if isinstance(model_or_model_url, torch.nn.Module):
                    logger.info(f"Trying to load from torch model")
                    __framework = "pytorch"
                    assert category is not None, "Category for inputs must be provided, when loading model manually"

                    model_key = model_key

                    # initialise all the parsers
                    image_parser = ImageParser(post_proc_fn=lambda x: torch.from_numpy(x).float())
                    text_parser = TextParser(tokenizer=tokenizer, post_proc_fn=lambda x: torch.from_numpy(x).int())

                    if isinstance(category, dict):
                        assert all([v in ["image", "text"] for v in category.values()])
                    else:
                        if category not in ["image", "text"]:
                            raise ValueError(f"Category: {category} is not supported yet. Raise a PR!")

                    if category == "text":
                        assert tokenizer != None, "tokenizer cannot be none for a text model!"

                    done = True
            except NameError:
                pass

            try:
                if isinstance(model_or_model_url, onnxruntime.InferenceSession):
                    logger.info(f"Trying to load from onnx model: {model_or_model_url}")
                    __framework = "onnx"

                    # we have to create templates using the nbox_meta
                    templates = None
                    if nbox_meta is not None:
                        all_inputs = nbox_meta["metadata"]["inputs"]
                        templates = {}
                        for node, meta in all_inputs.items():
                            templates[node] = [int(x["size"]) for x in meta["tensorShape"]["dim"]]

                    image_parser = ImageParser(post_proc_fn=lambda x: x.astype(np.float32), templates = templates)
                    text_parser = TextParser(tokenizer=tokenizer, post_proc_fn=lambda x: x.astype(np.int32))

                    self.session = model_or_model_url
                    self.input_names = [x.name for x in self.session.get_inputs()]
                    self.output_names = [x.name for x in self.session.get_outputs()]

                    logger.info(f"Inputs: {self.input_names}")
                    logger.info(f"Outputs: {self.output_names}")

                    done = True
            except NameError:
                pass

            if not done:
                raise ValueError(f"Unsupported model type: {type(model_or_model_url)}")


        # values coming from the blocks above
        self.model_or_model_url = model_or_model_url
        self.__framework = __framework
        self.text_parser = text_parser
        self.image_parser = image_parser
        self.nbox_meta = nbox_meta
        self.nbx_api_key = nbx_api_key
        self.category = category
        self.tokenizer = tokenizer
        self.model_key = model_key
        self.verbose = verbose
        self.__device = "cpu"
        self.cache_dir = gettempdir() if cache_dir == None else cache_dir

        logger.info(f"Model loaded successfully")
        logger.info(f"Model framework: {self.__framework}")
        logger.info(f"Model category: {self.category}")

    def fetch_meta_from_nbx_cloud(self):
        """When this is on NBX-Deploy cloud, fetch the metadata from the deployment node."""
        logger.info("Getting model metadata")
        URL = secret.get("nbx_url")
        r = requests.get(f"{URL}/api/model/get_model_meta", params=f"url={self.model_or_model_url}&key={self.nbx_api_key}")
        try:
            r.raise_for_status()
        except Exception as e:
            raise ValueError(f"Could not fetch metadata, please check status: {r.status_code}")

        # start getting the metadata, note that we have completely dropped using OVMS meta and instead use nbox_meta
        content = json.loads(r.content.decode())["meta"]
        nbox_meta = content["nbox_meta"]

        all_inputs = nbox_meta["metadata"]["inputs"]
        templates = {}
        for node, meta in all_inputs.items():
            templates[node] = [int(x["size"]) for x in meta["tensorShape"]["dim"]]

        if self.verbose:
            print("--------------")
            pp(nbox_meta)
            print("--------------")
            pp(templates)
            print("--------------")

        logger.info("Cloud infer metadata obtained")

        return nbox_meta, templates

    def __call__(self, input_object, return_inputs=False, return_dict=False, method=None, sklearn_args=None):
        r"""Caller is the most important UI/UX. The ``input_object`` can be anything from
        a tensor, an image file, filepath as string, string and must be processed automatically by a
        well written ``nbox.parser.BaseParser`` object . This ``__call__`` should understand the different
        usecases and manage accordingly.

        The current idea is that what ever the input, based on the category (image, text, audio, smell)
        it will be parsed through dedicated parsers that can make ingest anything.

        The entire purpose of this package is to make inference chill.

        Args:
            input_object (Any): input to be processed
            return_inputs (bool, optional): whether to return the inputs or not. Defaults to False.
            method(str, optional): specifically for sklearn models, this is the method to be called
                if nothing is provided then we call ``.predict()`` method.

        Returns:
            Any: currently this is output from the model, so if it is tensors and return dicts.
        """

        model_input = self._handle_input_object(input_object=input_object)

        if self.__framework == "nbx":
            logger.info(f"Hitting API: {self.model_or_model_url}")
            st = time()
            # OVMS has :predict endpoint and nbox has /predict
            _p = "/" if "export_type" in self.nbox_meta["spec"] else ":"
            json = {"inputs": model_input}
            if "export_type" in self.nbox_meta["spec"]:
                json["method"] = method
            r = requests.post(self.model_url + f"{_p}predict", json=json, headers={"NBX-KEY": self.nbx_api_key})
            et = time() - st
            out = None

            try:
                r.raise_for_status()
                out = r.json()

                # first try outputs is a key and we can just get the structure from the list
                if isinstance(out["outputs"], dict):
                    out = {k: np.array(v) for k, v in r.json()["outputs"].items()}
                elif isinstance(out["outputs"], list):
                    out = np.array(out["outputs"])
                else:
                    raise ValueError(f"Outputs must be a dict or list, got {type(out['outputs'])}")
                logger.info(f"Took {et:.3f} seconds!")
            except Exception as e:
                logger.info(f"Failed: {str(e)} | {r.content.decode()}")

        elif self.__framework == "sklearn" or self.__framework == "self-pkl":
            if "sklearn.neighbors.NearestNeighbors" in str(type(self.model_or_model_url)):
                method = getattr(self.model_or_model_url, "kneighbors") if method == None else getattr(self.model_or_model_url, method)
                out = method(model_input, **sklearn_args)
            elif "sklearn.cluster" in str(type(self.model_or_model_url)):
                if any(
                    x in str(type(self.model_or_model_url)) for x in ["AgglomerativeClustering", "DBSCAN", "OPTICS", "SpectralClustering"]
                ):
                    method = getattr(self.model_or_model_url, "fit_predict")
                    out = method(model_input)
            else:
                try:
                    method = getattr(self.model_or_model_url, "predict") if method == None else getattr(self.model_or_model_url, method)
                    out = method(model_input)
                except Exception as e:
                    print("[ERROR] Model Prediction Function is not yet registered " + e)

        elif self.__framework == "pytorch":
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

            # if self.nbox_meta is not None and self.nbox_meta.get("metadata", False) and self.nbox_meta["metadata"].get("outputs", False):
            #     outputs = self.nbox_meta["metadata"]["outputs"]
            #     if not isinstance(out, torch.Tensor):
            #         assert len(outputs) == len(out)
            #         out = {k: v.numpy() for k, v in zip(outputs, out)}
            #     else:
            #         out = {k: v.numpy() for k, v in zip(outputs, [out])}

        elif self.__framework == "onnx":
            if set(model_input.keys()) != set(self.input_names):
                diff = set(model_input.keys()) - set(self.input_names)
                return f"model_input keys do not match input_name: {diff}"
            out = self.model_or_model_url.run(self.output_names, model_input)
 
        # convert to dictionary if needed
        if return_dict:
            output_names = list(self.nbox_meta["metadata"]["outputs"].keys())
            if isinstance(out, (tuple, list)):
                out = {k: v for k, v in zip(output_names, out)}
            elif isinstance(out, dict):
                pass
            else:
                try:
                    if isinstance(out, (torch.Tensor, np.ndarray)):
                        out = {k: v.tolist() for k, v in zip(output_names, [out])}
                except NameError:
                    pass

                raise ValueError(f"Outputs must be a dict or list, got {type(out['outputs'])}")

        if return_inputs:
            return out, model_input
        return out

    def _handle_input_object(self, input_object):
        """First level handling to convert the input object to a fixed object"""
        # in case of scikit learn user must ensure that the input_object is model_input
        if self.__framework == "sklearn":
            return input_object

        elif self.__framework in ["nbx", "onnx"]:
            # the beauty is that the server is using the same code as this meaning that client
            # can get away with really simple API calls
            inputs_deploy = set(self.nbox_meta["metadata"]["inputs"].keys())
            if isinstance(input_object, dict):
                inputs_client = set(input_object.keys())
                assert inputs_deploy == inputs_client, f"Inputs mismatch, deploy: {inputs_deploy}, client: {inputs_client}"
                input_object = input_object
            else:
                if len(inputs_deploy) == 1:
                    input_object = {list(inputs_deploy)[0]: input_object}
                else:
                    assert len(input_object) == len(inputs_deploy), f"Inputs mismatch, deploy: {inputs_deploy}, client: {len(input_object)}"
                    input_object = {k: v for k, v in zip(inputs_deploy, input_object)}

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

        # Code below this part is super buggy and is useful for sklearn model,
        # please improve this as more usecases come up
        elif self.category == None:
            if isinstance(input_object, dict):
                return {k: v.tolist() for k, v in input_object.items()}
            return input_object.tolist()

        # when user gives a list as an input, it's better just to pass it as is
        # but when the input becomes a dict, this might fail.
        return input_object

    def get_nbox_meta(self, input_object, return_kwargs = True):
        """Get the nbox meta and trace args for the model with the given input object

        Args:
            input_object (Any): input to be processed
        """
        # this function gets the nbox metadata for the the current model, based on the input_object
        if self.__framework == "nbx":
            return self.nbox_meta

        args = None
        if self.__framework == "pytorch":
            args = inspect.getfullargspec(self.model_or_model_url.forward).args
            args.remove("self")

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
            input_names = tuple(["input_0"]) if args is None else tuple(args)
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
        if return_kwargs:
            return meta, out
        return meta

    def serialise(
        self,
        input_object,
        model_name=None,
        export_type="onnx",
        generate_ov_args = False,
        return_meta = False
    ):
        """This creates a singular .nbox file that contains the model binary and config file in ``self.cache_dir``

        Creates a folder at ``/tmp/{hash}`` and then adds three files to it:
            - ``/tmp/{hash}/{model_name}.{export_type}``
            - ``/tmp/{hash}/{model_name}.json``
            - ``/tmp/{hash}/{model_name}.nbox``

        and then later deletes the ``{model_name}.{export_type}`` and ``{model_name}.json`` and returns
        ``{model_name}.nbox`` and if required the metadata

        Args:
            input_object (Any): input to be processed
            model_name (str, optional): name of the model
            export_type (str, optional): [description]. Defaults to "onnx".
            generate_ov_args (bool, optional): Return the CLI args to be passed for Intel OpenVino
        """

        # First Step: check the args and see if conditionals are correct or not
        def __check_conditionals():
            assert self.__framework != "nbx", "This model is already deployed on the cloud"
            assert export_type in ["onnx", "torchscript", "pkl"], "Export type must be onnx, torchscript or pickle"
            if self.__framework == "sklearn":
                assert export_type in ["onnx", "pkl"], f"Export type must be onnx or pkl | got {export_type}"
            if self.__framework == "pytorch":
                assert export_type in ["onnx", "torchscript"], f"Export type must be onnx or torchscript | got {export_type}"

        # perform sanity checks on the input values
        __check_conditionals()

        # get metadata and exporing values
        nbox_meta, export_kwargs = self.get_nbox_meta(input_object, return_kwargs=True)
        _m_hash = utils.hash_(self.model_key)
        export_folder = utils.join(self.cache_dir, _m_hash)
        logger.info(f"Exporting model to {export_folder}")
        os.makedirs(export_folder, exist_ok=True)

        # intialise the console logger
        model_name = model_name if model_name is not None else f"{utils.get_random_name()}-{_m_hash[:4]}".replace("-", "_")
        export_model_path = utils.join(export_folder, f"{model_name}.{export_type}")
        logger.info("-" * 30 + f" Exporting {model_name}")

        # convert the model -> create a the spec, get the actual method for conversion
        logger.info(f"model_name: {model_name}")
        logger.info(f"Export type: {export_type}")
        
        try:
            export_fn = getattr(globals()[f"frm_{self.__framework}"], f"export_to_{export_type}", None)
            if export_fn == None:
                raise KeyError(f"Export type {export_type} not supported for {self.__framework}")
        except KeyError:
            raise ValueError(f"Framework {self.__framework} not supported (likely missing packages), check logs for more info")

        logger.info(f"Converting using: {export_fn}")
        export_fn(model=self.model_or_model_url, export_model_path=export_model_path, **export_kwargs)
        logger.info("Conversion Complete")

        # construct the output
        convert_args = None
        if generate_ov_args:
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
            logger.info(convert_args)

        # write the nbox meta in a file and tarball the model and meta
        nbox_meta = {
            "metadata": nbox_meta,
            "spec": {
                "category": self.category,
                "model_key": self.model_key,
                "model_name": model_name,
                "src_framework": self.__framework,
                "export_type": export_type,
                "convert_args": convert_args
            },
        }
        meta_path = utils.join(export_folder, f"{model_name}.json")
        with open(meta_path, "w") as f:
            f.write(json.dumps(nbox_meta))

        nbx_path = os.path.join(export_folder, f"{model_name}.nbox")
        with tarfile.open(nbx_path, "w|gz") as tar:
            tar.add(export_model_path, arcname=os.path.basename(export_model_path))
            tar.add(meta_path, arcname=os.path.basename(meta_path))

        # remove the files
        os.remove(export_model_path)
        os.remove(meta_path)

        if return_meta:
            return nbx_path, nbox_meta
        return nbx_path

    @classmethod
    def deserialise(cls, filepath, verbose=False):
        """This is the counterpart of ``serialise``, it deserialises the model from the .nbox file.

        Args:
            nbx_path (str): path to the .nbox file
        """
        if not tarfile.is_tarfile(filepath) or not filepath.endswith(".nbox"):
            raise ValueError(f"{filepath} is not a valid .nbox file")

        with tarfile.open(filepath, "r:gz") as tar:
            folder = utils.join(gettempdir(), os.path.split(filepath)[-1].split(".")[0])
            tar.extractall(folder)

        files = glob(utils.join(folder, "*"))
        nbox_meta, nbox_meta_path, model_path = None, None, None
        for f in files:
            ext = f.split(".")[-1]
            if ext == "json":
                nbox_meta_path = f
                with open(f, "r") as f:
                    nbox_meta = json.load(f)
            elif ext in ["onnx", "pkl", "pt", "torchscript"]:
                model_path = f

        logger.info(f"nbox_meta: {nbox_meta_path}")
        logger.info(f"model_path: {model_path}")
        if not (nbox_meta or model_path):
            shutil.rmtree(folder)
            raise ValueError(f"{filepath} is not a valid .nbox file")

        export_type = nbox_meta["spec"]["export_type"]
        src_framework = nbox_meta["spec"]["src_framework"]
        category = nbox_meta["spec"]["category"]

        try:
            if export_type == "onnx":
                model = onnxruntime.InferenceSession(model_path)
            elif src_framework == "pt":
                if export_type == "torchscript":
                    model = torch.jit.load(model_path, map_location="cpu")
            elif src_framework == "sk":
                if export_type == "pkl":
                    with open(model_path, "rb") as f:
                        model = joblib.load(f)
        except Exception as e:
            raise ValueError(f"{export_type} not supported, are you missing packages? {e}")

        model = cls(model_or_model_url=model, category=category, nbox_meta=nbox_meta, verbose=verbose)
        shutil.rmtree(folder)
        return model

    def deploy(
        self,
        input_object,
        model_name=None,
        wait_for_deployment=False,
        runtime="onnx",
        deployment_type="nbox",
        deployment_id=None,
        deployment_name=None,
    ):
        """NBX-Deploy `read more <https://nimbleboxai.github.io/nbox/nbox.model.html>`_

        This deploys the current model onto our managed K8s clusters. This tight product service integration
        is very crucial for us and is the best way to make deploy a model for usage.

        Raises appropriate assertion errors for strict checking of inputs

        Args:
            input_object (Any): input to be processed
            model_name (str, optional): custom model name for this model
            cache_dir (str, optional): custom caching directory
            wait_for_deployment (bool, optional): wait for deployment to complete
            runtime (str, optional): runtime to use for deployment should be one of ``["onnx", "torchscript"]``, default is ``onnx``
            deployment_type (str, optional): deployment type should be one of ``['ovms2', 'nbox']``, default is ``nbox``
            deployment_id (str, optional): ``deployment_id`` to put this model under, if you do not pass this
                it will automatically create a new deployment check `platform <https://nimblebox.ai/oneclick>`_
                for more info or check the logs.
            deployment_name (str, optional): if ``deployment_id`` is not given and you want to create a new
                deployment group (ie. webserver will create a new ``deployment_id``) you can tell what name you
                want, be default it will create a random name.
        """
        # First Step: check the args and see if conditionals are correct or not
        def __check_conditionals():
            assert self.__framework != "nbx", "This model is already deployed on the cloud"
            assert deployment_type in ["ovms2", "nbox"], f"Only OpenVino and Nbox-Serving is supported got: {deployment_type}"
            if self.__framework == "sklearn":
                assert deployment_type == "nbox", "Only ONNX Runtime is supported for scikit-learn Framework"
            if deployment_type == "ovms2":
                assert runtime == "onnx", "Only ONNX Runtime is supported for OVMS2 Framework"
            if deployment_id and deployment_name:
                raise ValueError("deployment_id and deployment_name cannot be used together, pass only id")

        # perform sanity checks on the input values
        __check_conditionals()

        export_model_path, nbox_meta = self.serialise(
            input_object,
            model_name = model_name,
            export_type = runtime,
            generate_ov_args = deployment_type == "ovms2",
            return_meta = True
        )

        nbox_meta["spec"]["deployment_type"] = deployment_type
        nbox_meta["spec"]["deployment_id"] = deployment_id
        nbox_meta["spec"]["deployment_name"] = deployment_name

        # OCD baby!
        return deploy_model(
            export_model_path=export_model_path,
            nbox_meta=nbox_meta,
            wait_for_deployment=wait_for_deployment,
        )
