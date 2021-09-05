# singluar loader file for all models in nbox

import os
import re
import json
from typing import Dict

import inspect
import warnings

from nbox.model import Model
from nbox.api import NBXApi
from nbox.utils import is_available

import torch

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
#   # "key": (builder_function, "task_type", "source", "pre", "task", "post")
# }
#
# Structure of each loader function looks as follows:
# def loader_fn() -> <dict as above>
#
# Each model builder function looks as follows:
# def model_builder() -> (model, model_kwargs)


def load_efficientnet_pytorch_models(pop_kwargs=["model_instr"]) -> Dict:
    import efficientnet_pytorch

    def model_builder(pretrained=False, **kwargs):
        if pretrained:
            model_fn = efficientnet_pytorch.EfficientNet.from_pretrained
        else:
            model_fn = efficientnet_pytorch.EfficientNet.from_name

        kwargs = remove_kwargs(pop_kwargs, **kwargs)
        return model_fn(**kwargs), {}

    return {"efficientnet_pytorch/efficientnet": (model_builder, "image")}


def load_torchvision_models(pop_kwargs=["model_instr"]) -> Dict:
    import torchvision

    def model_builder(model, pretrained=False, **kwargs):
        model_fn = {
            "alexnet": torchvision.models.alexnet,
            "resnet18": torchvision.models.resnet18,
            "resnet34": torchvision.models.resnet34,
            "resnet50": torchvision.models.resnet50,
            "resnet101": torchvision.models.resnet101,
            "resnet152": torchvision.models.resnet152,
            "vgg11": torchvision.models.vgg11,
            "vgg11-bn": torchvision.models.vgg11_bn,
            "vgg13": torchvision.models.vgg13,
            "vgg13-bn": torchvision.models.vgg13_bn,
            "vgg16": torchvision.models.vgg16,
            "vgg16-bn": torchvision.models.vgg16_bn,
            "vgg19": torchvision.models.vgg19,
            "vgg19-bn": torchvision.models.vgg19_bn,
            "squeezenet1": torchvision.models.squeezenet1_0,
            "squeezenet1-1": torchvision.models.squeezenet1_1,
            "densenet121": torchvision.models.densenet121,
            "densenet161": torchvision.models.densenet161,
            "densenet169": torchvision.models.densenet169,
            "densenet201": torchvision.models.densenet201,
            "inceptionv3": torchvision.models.inception_v3,
            "googlenet": torchvision.models.googlenet,
            "shufflenetv2-0.5": torchvision.models.shufflenet_v2_x0_5,
            "shufflenetv2-1.0": torchvision.models.shufflenet_v2_x1_0,
            "shufflenetv2-1.5": torchvision.models.shufflenet_v2_x1_5,
            "shufflenetv2-2.0": torchvision.models.shufflenet_v2_x2_0,
            "mobilenetv2": torchvision.models.mobilenet_v2,
            "mobilenetv3-small": torchvision.models.mobilenet_v3_small,
            "mobilenetv3-large": torchvision.models.mobilenet_v3_large,
            "resnext50": torchvision.models.resnext50_32x4d,
            "resnext101": torchvision.models.resnext101_32x8d,
            "wide-resnet50": torchvision.models.wide_resnet50_2,
            "wide-resnet101": torchvision.models.wide_resnet101_2,
            "mnasnet0-5": torchvision.models.mnasnet0_5,
            "mnasnet0-75": torchvision.models.mnasnet0_75,
            "mnasnet1-0": torchvision.models.mnasnet1_0,
            "mnasnet1-3": torchvision.models.mnasnet1_3,
            "fasterrcnn_mobilenet_v3_large_fpn": torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn,
            "fasterrcnn_mobilenet_v3_large_320_fpn": torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn,
            "fasterrcnn_resnet50_fpn": torchvision.models.detection.fasterrcnn_resnet50_fpn,
            "maskrcnn_resnet50_fpn": torchvision.models.detection.maskrcnn_resnet50_fpn,
            "keypointrcnn_resnet50_fpn": torchvision.models.detection.keypointrcnn_resnet50_fpn,
            "retinanet_resnet50_fpn": torchvision.models.detection.retinanet_resnet50_fpn,
            "mc3_18": torchvision.models.video.mc3_18,
            "r3d_18": torchvision.models.video.r3d_18,
            "r2plus1d_18": torchvision.models.video.r2plus1d_18,
            "deeplabv3_mobilenet_v3_large": torchvision.models.segmentation.deeplabv3_mobilenet_v3_large,
            "deeplabv3_resnet101": torchvision.models.segmentation.deeplabv3_resnet101,
            "deeplabv3_resnet50": torchvision.models.segmentation.deeplabv3_resnet50,
            "fcn_resnet50": torchvision.models.segmentation.fcn_resnet50,
            "fcn_resnet101": torchvision.models.segmentation.fcn_resnet101,
            "lraspp_mobilenet_v3_large": torchvision.models.segmentation.lraspp_mobilenet_v3_large,
        }.get(model, None)
        if model_fn == None:
            raise IndexError(f"Model: {model} not found in torchvision")

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
    import transformers

    def hf_model_builder(model, model_instr, **kwargs):
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

        # TODO: @yashbonde remove GPT hardcoded dependency
        if not tokenizer.pad_token_id:
            tokenizer.pad_token = "<|endoftext|>"
        model = _auto_loaders[auto_model_type].from_pretrained(model, **kwargs)
        # ignoring task for now
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


def load(model_key: str = None, nbx_api_key: str = None, cloud_infer: bool = False, **loader_kwargs):
    """Returns nbox.Model from a model (key), can optionally setup a connection to
    cloud inference on a Nimblebox instance.

    Args:
        model_key (str, optional): key for which to load the model, the structure looks as follows:
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
    # step 1: check the model key if it is a file path, declare such a variable
    if os.path.exists(model_key):
        model_path = os.path.abspath(model_key)
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

        model = torch.jit.load(model_path, map_location = "cpu")
        out = Model(model, category=category, tokenizer=tokenizer, model_key=spec["model_key"], model_meta = model_meta)

    else:
        # the input key can also contain instructions on how to run a particular models and so
        model_key_parts = re.findall(model_key_regex, model_key)
        if not model_key_parts:
            raise ValueError(f"Key: {model_key} incorrect, please check once!")

        # this key is valid, now get it's components
        model_key, src, src_key, model_instr = model_key_parts[0]
        if src not in PT_SOURCES:
            raise ValueError(f"Model source: {src} not found. Is this package installed!")
        model_fn, model_meta = PRETRAINED_MODELS.get(model_key, (None, None))
        if model_meta is None:
            model_fn, model_meta = PRETRAINED_MODELS.get(src, (None, None))
            if model_meta is None:
                raise IndexError(f"Model: {model_key} not found")

        # load the model based on local infer or cloud infer
        if cloud_infer and nbx_api_key:
            out = NBXApi(model_key_or_url=model_key, category=model_meta, nbx_api_key=nbx_api_key)
        else:
            model, model_kwargs = model_fn(model=src_key, model_instr=model_instr, **loader_kwargs)
            out = Model(model=model, category=model_meta, model_key=model_key, model_meta = None, **model_kwargs)

    return out
