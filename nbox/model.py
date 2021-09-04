import os
from re import I
from numpy.lib.arraysetops import isin
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
    """single unified Image parser that consumes different types of data
    and returns a processed numpy array"""

    def __init__(self, pre_proc_fn=None):
        self.pre_proc_fn = pre_proc_fn

    # if the input is torch.int then rescale it -1, 1
    def rescale(self, x):
        if utils.is_available("torch") and isinstance(x, torch.Tensor) and "int" in str(x.dtype):
            return (x - 122.5) / 122.5
        return x

    def rearrange(self, x):
        if len(x.shape) == 3 and x.shape[0] != 3:
            return x.permute(2, 0, 1)
        elif len(x.shape) == 4 and x.shape[1] != 3:
            return x.permute(0, 3, 1, 2)
        return x

    def handle_string(self, x: str):
        if os.path.exists(x):
            utils.info(" - ImageParser - (A1)")
            out = np.array(Image.open(x))
            return out
        elif x.startswith("http:") or x.startswith("https:"):
            utils.info(" - ImageParser - (A2)")
            out = np.array(utils.get_image(x))
            return out
        else:
            utils.info(" - ImageParser - (A)")
            raise ValueError("Cannot process string that is not Image path")

    def handle_list(self, x: list):
        utils.info(" - ImageParser - (B)")
        return [self(i) for i in x]

    def handle_dictionary(self, x: dict):
        utils.info(" - ImageParser - (C)")
        return {k: self(v) for k, v in x.items()}

    def handle_pil_image(self, x: Image):
        utils.info(" - ImageParser - (E)")
        out = self(np.array(x))
        return out

    def __call__(self, x):
        if utils.is_available("torch") and isinstance(x, torch.Tensor):
            x = x
            x = self.pre_proc_fn(x) if self.pre_proc_fn is not None else x
            out = self.rearrange(self.rescale(x.unsqueeze(0)))
        elif isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
            x = self.pre_proc_fn(x) if self.pre_proc_fn is not None else x
            out = self.rearrange(self.rescale(x.unsqueeze(0)))
        elif isinstance(x, str):
            x = torch.from_numpy(self.handle_string(x))
            x = self.pre_proc_fn(x) if self.pre_proc_fn is not None else x
            out = self.rearrange(self.rescale(x.unsqueeze(0)))
        elif isinstance(x, list):
            out = self.handle_list(x)

            # if out is list of dicts then create a dict with concatenated tensors
            if isinstance(out[0], dict):
                # assert all keys are same in the list
                assert all([set(out[0].keys()) == set(i.keys()) for i in out])
                out = {k: torch.cat([i[k] for i in out]) for k in out[0].keys()}
            elif isinstance(out[0], torch.Tensor):
                out = torch.cat(out)
            else:
                raise ValueError("Unsupported type: {}".format(type(out[0])))

        elif isinstance(x, dict):
            x = self.handle_dictionary(x)
            x = {k: self.pre_proc_fn(v) if self.pre_proc_fn is not None else v for k, v in x.items()}
            out = {k: self.rescale(v) for k, v in x.items()}
            out = {k: self.rearrange(v) for k, v in x.items()}
        elif isinstance(x, Image.Image):
            x = self.handle_pil_image(x)
            out = self.rearrange(self.rescale(x))
        else:
            utils.info(" - ImageParser - (D)")
            raise ValueError(f"Cannot process item of dtype: {type(x)}")
        return out


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
        self.text_parser = TextParser(tokenizer=tokenizer)

        if isinstance(self.category, dict):
            assert all([v in ["image", "text"] for v in self.category.values()])
        else:
            if self.category not in ["image", "text"]:
                raise ValueError(f"Category: {self.category} is not supported yet. Raise a PR!")

        if self.category == "text":
            assert tokenizer != None, "tokenizer cannot be none for a text model!"

    def get_model(self):
        return self.model

    def eval(self):
        self.model.eval()

    def train(self):
        self.model.train()

    def __repr__(self):
        return f"<nbox.Model: {repr(self.model)} >"

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
            return input_dict

        if self.category == "image":
            input_obj = self.image_parser(input_object)
            return input_obj

        elif self.category == "text":
            # perform parsing for text and pass to the model
            input_dict = self.text_parser(input_object)
            input_dict = {k: torch.from_numpy(v) for k, v in input_dict.items()}
            return input_dict

    def __call__(self, input_object: Any, return_inputs=False, verbose=False):
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

        model_input = self._handle_input_object(input_object=input_object, verbose=verbose)

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
            input_shapes = tuple([tuple(v.shape) for k, v in model_input.items()])
        elif isinstance(model_input, torch.Tensor):
            args = tuple([model_input])
            input_names = tuple(["input_0"])
            input_shapes = tuple([tuple(model_input.shape)])
        dynamic_axes = {i: dynamic_axes_dict for i in input_names}

        if isinstance(model_output, dict):
            output_names = tuple(model_output.keys())
            output_shapes = tuple([tuple(v.shape) for k, v in model_output.keys()])
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

        # OCD baby!
        out = network.ocd(
            model_key=self.model_key,
            model=self.model,
            args=args,
            input_names=input_names,
            input_shapes=input_shapes,
            output_names=output_names,
            output_shapes=output_shapes,
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
