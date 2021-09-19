# singluar loader file for all models in nbox

import os
import re
import json
import inspect
import warnings
import requests
from typing import Dict

import torch

from nbox.model import Model
from nbox.utils import is_available, fetch

model_key_regex = re.compile(r"^((\w+)\/([\w\/-]+)):*([\w+:]+)?$")

# util functions
def remove_kwargs(pop_list, **kwargs):
    for p in pop_list:
        kwargs.pop(p)
    return kwargs


# --- model loader functions: add your things here
# guide: all models are indexed as follows
# {
#   "key": (builder_function, "category")
#
#   # to be moved to
#   "key": (builder_function, "task_type", "source", "pre", "task", "post")
# }
#
# Structure of each loader function looks as follows:
# def loader_fn() -> <dict as above>
#
# Each model builder function looks as follows:
# def model_builder() -> (model, model_kwargs)


def load_efficientnet_pytorch_models(pop_kwargs=["model_instr"]) -> Dict:
    def model_builder(pretrained=False, **kwargs):
        import efficientnet_pytorch

        if pretrained:
            model_fn = efficientnet_pytorch.EfficientNet.from_pretrained
        else:
            model_fn = efficientnet_pytorch.EfficientNet.from_name

        kwargs = remove_kwargs(pop_kwargs, **kwargs)
        return model_fn(**kwargs), {}

    return {"efficientnet_pytorch/efficientnet": (model_builder, "image")}


def load_torchvision_models(pop_kwargs=["model_instr"]) -> Dict:
    def model_builder(model, pretrained=False, **kwargs):
        # tv_mr = torchvision models registry
        # this is a json object that maps all the models to their respective methods from torchvision
        # the trick to loading is to eval the method string
        tv_mr = json.loads(fetch("https://raw.githubusercontent.com/NimbleBoxAI/nbox/master/assets/pt_models.json").decode("utf-8"))
        model_fn = tv_mr.get(model, None)
        if model_fn == None:
            raise IndexError(f"Model: {model} not found in torchvision")

        # torchvision is imported here, if it is imported in the outer method, eval fails
        # reason: unknown
        import torchvision

        model_fn = eval(model_fn)
        kwargs = remove_kwargs(pop_kwargs, **kwargs)

        # compare variables between the model_fn and kwargs if they are different then remove it with warning
        arg_spec = inspect.getfullargspec(model_fn)
        if kwargs and arg_spec.varkw != None:
            diff = set(kwargs.keys()) - set(arg_spec.args)
            for d in list(diff):
                warnings.warn(f"Ignoring unknown argument: {d}")
                kwargs.pop(d)

        model = model_fn(pretrained=pretrained, **kwargs)
        return model, {}

    return {"torchvision": (model_builder, "image")}


def load_transformers_models() -> Dict:
    def hf_model_builder(model, model_instr, **kwargs):
        import transformers

        _auto_loaders = {x: getattr(transformers, x) for x in dir(transformers) if x[:4] == "Auto" and x != "AutoConfig"}

        model_instr = model_instr.split("::")
        if len(model_instr) == 1:
            auto_model_type = model_instr[0]
        else:
            # if the task is given, validate that as well
            auto_model_type, task = model_instr
            assert task in [
                "generation",
                "masked_lm",
            ], "For now only the following are supported: `generation`, `masked_lm`"

        # initliase the model and tokenizer object
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, **kwargs)

        # tokenizer.pad_token = tokenizer.eos_token if not tokenizer.pad_token_id else tokenizer.pad_token
        model = _auto_loaders[auto_model_type].from_pretrained(model, **kwargs)

        # return the model and tokenizer
        return model, {"tokenizer": tokenizer}

    return {"transformers": (hf_model_builder, "text")}


### ----- pretrained models master index
# add code based on conditionals, best way is to only include those that
# have proper model building code like transformers, torchvision, etc.

PRETRAINED_MODELS = {}
all_repos = ["efficientnet_pytorch", "torchvision", "transformers"]

for repo in all_repos:
    if is_available(repo):
        print(f"Loading pretrained models from {repo}")
        PRETRAINED_MODELS.update(locals()[f"load_{repo}_models"]())

# if there are no pretrained models available, then raise an error
if not PRETRAINED_MODELS:
    raise ValueError("No pretrained models available. Please install PyTorch or torchvision or transformers to use pretrained models.")


PT_SOURCES = list(set([x.split("/")[0] for x in PRETRAINED_MODELS]))


# ---- load function has to manage everything and return Model object properly initialised


def load(model_key_or_url: str = None, nbx_api_key: str = None, verbose=False, **loader_kwargs):
    """Returns nbox.Model from a model (key), can optionally setup a connection to
    cloud inference on a Nimblebox instance.

    Args:
        model_key_or_url (str, optional): key for which to load the model, the structure looks as follows:
            ```
            source/(source/key)::<pre::task::post>
            ```
            Defaults to None.
        nbx_api_key (str, optional): Your Nimblebox API key. Defaults to None.
        cloud_infer (bool, optional): If true uses Nimblebox deployed inference and logs in
            using `nbx_api_key`. Defaults to False.

    Raises:
        ValueError: If `source` is not found
        IndexError: If `source` is found but `source/key` is not found

    Returns:
        nbox.Model: when using local inference
        nbox.NBXApi: when using cloud inference
    """
    # check the model key if it is a file path, then check if
    if os.path.exists(model_key_or_url):
        model_path = os.path.abspath(model_key_or_url)
        model_meta_path = ".".join(model_path.split(".")[:-1] + ["json"])
        assert os.path.exists(model_meta_path), f"Model meta file not found: {model_meta_path}"

        with open(model_meta_path, "r") as f:
            model_meta = json.load(f)
            spec = model_meta["spec"]

        category = spec["category"]
        tokenizer = None
        if category == "text":
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model_path)

        model = torch.jit.load(model_path, map_location="cpu")
        out = Model(model, category=category, tokenizer=tokenizer, model_key=spec["model_key"], model_meta=model_meta, verbose=verbose)

    # if this is a nbx-deployed model
    elif model_key_or_url.startswith("http"):
        out = Model(model_or_model_url=model_key_or_url, nbx_api_key=nbx_api_key, verbose=verbose)

    else:
        # the input key can also contain instructions on how to run a particular models and so
        model_key_parts = re.findall(model_key_regex, model_key_or_url)
        if not model_key_parts:
            raise ValueError(f"Key: {model_key_or_url} incorrect, please check!")

        # this key is valid, now get it's components
        model_key, src, src_key, model_instr = model_key_parts[0]
        if src not in PT_SOURCES:
            raise ValueError(f"Model source: {src} not found. Is this package installed!")

        # sometimes you'll find the whole key, sometimes from the source, so check both
        model_fn, model_meta = PRETRAINED_MODELS.get(model_key, (None, None))
        if model_meta is None:
            model_fn, model_meta = PRETRAINED_MODELS.get(src, (None, None))
            if model_meta is None:
                raise IndexError(f"Model: {model_key} not found")

        # now just load the underlying graph and the model and off you go
        model, model_kwargs = model_fn(model=src_key, model_instr=model_instr, **loader_kwargs)
        out = Model(model_or_model_url=model, category=model_meta, model_key=model_key, model_meta=None, verbose=verbose, **model_kwargs)

    return out
