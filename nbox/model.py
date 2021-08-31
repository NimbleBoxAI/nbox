import os
import torch
import numpy as np
from PIL import Image

from typing import Any

from nbox import network
from nbox import processing
from nbox import utils

# ----- parsers
# These objects are mux, they consume and streamline the output
# Don't know what mux are? Study electronics.


class ImageParser:
    """single unified Image parser that returns PIL.Image objects by consuming multiple differnet data-types"""

    def handle_string(self, x: str):
        if os.path.exists(x):
            utils.info(" - ImageParser - (A1)")
            return [Image.open(x)]
        if x.startswith("http:") or x.startswith("https:"):
            utils.info(" - ImageParser - (A2)")
            return [utils.get_image(x)]
        else:
            utils.info(" - ImageParser - (B)")
            raise ValueError("Cannot process string that is not Image path")

    def handle_numpy(self, obj):
        if obj.dtype == np.float32 or obj.dtype == np.float64:
            utils.info(" - ImageParser - (C)")
            obj *= 122.5
            obj += 122.5
        utils.info(" - ImageParser - (C2)")
        if obj.dtype != np.uint8:
            obj = obj.astype(np.uint8)
        return [Image.fromarray(obj)]

    def handle_torch_tensor(self, obj):
        if obj.dtype == torch.float:
            utils.info(" - ImageParser - (C)")
            obj *= 122.5
            obj += 122.5
            obj = obj.numpy()
        else:
            raise ValueError(f"Incorrect datatype for torch.tensor: {obj.dtype}")
        utils.info(" - ImageParser - (C2)")
        if obj.dtype != np.uint8:
            obj = obj.astype(np.uint8)
        return [Image.fromarray(obj)]

    def __call__(self, x):
        if isinstance(x, str):
            proc_fn = self.handle_string
        elif isinstance(x, np.ndarray):
            proc_fn = self.handle_numpy
        elif isinstance(x, Image.Image):
            utils.info(" - ImageParser - (D)")
            proc_fn = lambda x: [x]
        else:
            utils.info(" - ImageParser - (E)")
            raise ValueError(f"Cannot process item of dtype: {type(x)}")
        return proc_fn(x)


class TextParser:
    """Unified Text parsing engine, returns a list of dictionaries"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def handle_string(self, x):
        utils.info(" - TextParser - (A)")
        return {k: v.numpy() for k, v in self.tokenizer(x, return_tensors="pt").items()}

    def handle_numpy(self, x):
        utils.info(" - TextParser - (B)")
        if x.dtype != np.int32 and x.dtype != np.int64:
            raise ValueError(f"Incorrect datatype for np.array: {x.dtype} | {x.dtype == np.int64}")
        return {"input_ids": x}

    def handle_dict(self, x):
        _x = {}
        utils.info(" - TextParser - (C)")
        for k, v in x.items():
            if isinstance(v, (list, tuple)) and isinstance(v[0], (list, tuple)):
                # list of lists -> batch processing
                utils.info(" - TextParser - (C1)")
                seqlen = len(v[0])
                assert len(v) * seqlen == sum([len(x) for x in v]), "Inconsistent values in list sequences"
                _x[k] = np.array(v).astype(np.int32)
            elif isinstance(v, (list, tuple)) and isinstance(v[0], int):
                utils.info(" - TextParser - (C2)")
                _x[k] = np.array(v).astype(np.int32)
            else:
                raise ValueError("Cannot parse dict items")
        return _x

    def handle_list_tuples(self, x):
        utils.info(" - TextParser - (D)")
        if isinstance(x[0], int):
            utils.info(" - TextParser - (D1)")
        elif isinstance(x[0], list):
            utils.info(" - TextParser - (D2)")
            seqlen = len(x[0])
            assert len(x) * seqlen == sum([len(y) for y in x]), "Inconsistent values in list sequences"
        elif isinstance(x[0], str):
            utils.info(" - TextParser - (D3)")
            if self.tokenizer == None:
                raise ValueError("self.tokenizer cannot be None when string input")
            tokens = self.tokenizer(x, padding="longest", return_tensors="pt")
            return {k: v.numpy().astype(np.int32) for k, v in tokens.items()}
        else:
            raise ValueError(f"Cannot parse list of item: {type(x[0])}")
        return {"input_ids": np.array(x).astype(np.int32)}

    def handle_torch_tensor(self, x):
        utils.info(" - Text Parser - (F)")
        assert x.dtype == torch.long, f"Incorrect datatype for torch.tensor: {x.dtype}"
        return {"input_ids": x}

    def __call__(self, x):
        if isinstance(x, str):
            proc_fn = self.handle_string
        elif isinstance(x, np.ndarray):
            proc_fn = self.handle_numpy
        elif isinstance(x, dict):
            proc_fn = self.handle_dict
        elif isinstance(x, (list, tuple)):
            proc_fn = self.handle_list_tuples
        elif utils.is_available("torch") and isinstance(x, torch.Tensor):
            proc_fn = self.handle_torch_tensor
        else:
            utils.info(" - ImageParser - (E)")
            raise ValueError(f"Cannot process item of dtype: {type(x)}")
        return proc_fn(x)


class Model:
    def __init__(self, model: torch.nn.Module, category: str, tokenizer=None, model_key: str = None):
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
        self.model = model
        self.category = category
        self.model_key = model_key

        # initialise all the parsers, like WTH, how bad would it be
        self.image_parser = ImageParser()
        self.text_parser = None

        if self.category not in ["image", "text"]:
            raise ValueError(f"Category: {self.category} is not supported yet. Raise a PR!")

        if self.category == "text":
            assert tokenizer != None, "tokenizer cannot be none for a text model!"
            self.text_parser = TextParser(tokenizer=tokenizer)

    def get_model(self):
        return self.model

    def eval(self):
        self.model.eval()

    def train(self):
        self.model.train()

    def __repr__(self):
        return f"<nbox.Model: {repr(self.model)} >"

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
                input_tensor = processing.totensor(pil_img)
            return input_tensor

        elif self.category == "text":
            # perform parsing for text and pass to the model
            input_dict = self.text_parser(input_object)
            input_dict = {k: torch.from_numpy(v) for k, v in input_dict.items()}
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

        Returns:
            Any: currently this is output from the model, so if it is tensors and return dicts.
        """

        model_input = self._handle_input_object(input_object=input_object)

        if isinstance(model_input, dict):
            out = self.model(**model_input)
        else:
            assert isinstance(model_input, torch.Tensor)
            out = self.model(model_input)

        if return_inputs:
            return out, model_input
        return out

    def deploy(self, input_object: Any, username: str = None, password: str = None, model_name: str = None, cache_dir: str = None):
        """OCD your model on NBX platform.

        Args:
            input_object (Any): input to be processed
            username (str, optional): your username, ignore if on NBX platform. Defaults to None.
            password (str, optional): your password, ignore if on NBX platform. Defaults to None.
            model_name (str, optional): custom model name for this model. Defaults to None.
            cache_dir (str, optional): Custom caching directory. Defaults to None.

        Returns:
            (str, None): if deployment is successful then push then return the URL endpoint else return None
        """
        # user will always have to pass the input_object
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
        elif isinstance(model_input, torch.Tensor):
            args = tuple([model_input])
            input_names = tuple(["input_0"])
        dynamic_axes = {i: dynamic_axes_dict for i in input_names}

        if isinstance(model_output, dict):
            output_names = tuple(model_output.keys())
        elif isinstance(model_output, (list, tuple)):
            output_names = tuple([f"output_{i}" for i, x in enumerate(model_output)])
        elif isinstance(model_output, torch.Tensor):
            output_names = tuple(["output_0"])

        # OCD baby!
        out = network.ocd(
            model_key=self.model_key,
            model=self.model,
            args=args,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            category=self.category,
            username=username,
            password=password,
            model_name=model_name,
            cache_dir=cache_dir,
        )

        return out

    def export(self, folder_path):
        """Creates a FastAPI / Flask folder with all the things required to serve this model

        Args:
            folder_path (str): folder where to put things in

        Raises:
            NotImplementedError
        """
        raise NotImplementedError()
