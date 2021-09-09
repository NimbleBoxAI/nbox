# this file has the code for nbox.Model that is the holy grail of the project

import torch
from typing import Any

from nbox import network
from nbox.parsers import ImageParser, TextParser


class Model:
    def __init__(self, model: torch.nn.Module, category: str, tokenizer=None, model_key: str = None, model_meta: dict = None):
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
        self.model_meta = model_meta  # this is a big dictionary (~ same) as TF-Serving metadata

        # initialise all the parsers, like WTH, how bad would it be
        self.image_parser = ImageParser()
        self.text_parser = TextParser(tokenizer=tokenizer)

        if isinstance(self.category, dict):
            assert all([v in ["image", "text", None] for v in self.category.values()])
        else:
            if self.category not in ["image", "text", None]:
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

        elif self.category == "image":
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
            verbose (bool, optional): whether to print the inputs or not. Defaults to False.

        Returns:
            Any: currently this is output from the model, so if it is tensors and return dicts.
        """

        model_input = self._handle_input_object(input_object=input_object, verbose=verbose)

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

        spec = {"category": self.category, "model_key": self.model_key}

        # OCD baby!
        out = network.ocd(
            model_key=self.model_key,
            model=self.model,
            args=args,
            outputs=model_output,
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
            spec=spec,
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
