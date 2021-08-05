# this is too much code, quickly iterate and simplify the process!

from nbox import Model

def is_available(package: str):
    import importlib
    try:
        importlib.import_module(package)
        return True
    except ImportError as e:
        return False

# --- model loader functions: add your things here
# guide: all models are indexed as follows
# {
#   "key": (builder_function, "task_type", "source", "pre", "task", "post")
# }

def load_efficientnet_pytorch_models():
    import efficientnet_pytorch
    return {
        "efficientnet_pytorch/efficientnet_from_name": (efficientnet_pytorch.EfficientNet.from_name, "image"),
        "efficientnet_pytorch/efficientnet_pretrained": (efficientnet_pytorch.EfficientNet.from_pretrained, "image"),
    }


def load_torchvision_models():
    import torchvision
    return {
        "torchvision/alexnet": (torchvision.models.alexnet, "image"),
        "torchvision/resnet18": (torchvision.models.resnet18, "image"),
        "torchvision/resnet34": (torchvision.models.resnet34, "image"),
        "torchvision/resnet50": (torchvision.models.resnet50, "image"),
        "torchvision/resnet101": (torchvision.models.resnet101, "image"),
        "torchvision/resnet152": (torchvision.models.resnet152, "image"),
        "torchvision/vgg11": (torchvision.models.vgg11, "image"),
        "torchvision/vgg11-bn": (torchvision.models.vgg11_bn, "image"),
        "torchvision/vgg13": (torchvision.models.vgg13, "image"),
        "torchvision/vgg13-bn": (torchvision.models.vgg13_bn, "image"),
        "torchvision/vgg16": (torchvision.models.vgg16, "image"),
        "torchvision/vgg16-bn": (torchvision.models.vgg16_bn, "image"),
        "torchvision/vgg19": (torchvision.models.vgg19, "image"),
        "torchvision/vgg19-bn": (torchvision.models.vgg19_bn, "image"),
        "torchvision/squeezenet1": (torchvision.models.squeezenet1_0, "image"),
        "torchvision/squeezenet1-1": (torchvision.models.squeezenet1_1, "image"),
        "torchvision/densenet121": (torchvision.models.densenet121, "image"),
        "torchvision/densenet161": (torchvision.models.densenet161, "image"),
        "torchvision/densenet169": (torchvision.models.densenet169, "image"),
        "torchvision/densenet201": (torchvision.models.densenet201, "image"),
        "torchvision/inceptionv3": (torchvision.models.inception_v3, "image"),
        "torchvision/googlenet": (torchvision.models.googlenet, "image"),
        "torchvision/shufflenetv2-0.5": (torchvision.models.shufflenet_v2_x0_5, "image"),
        "torchvision/shufflenetv2-1.0": (torchvision.models.shufflenet_v2_x1_0, "image"),
        "torchvision/shufflenetv2-1.5": (torchvision.models.shufflenet_v2_x1_5, "image"),
        "torchvision/shufflenetv2-2.0": (torchvision.models.shufflenet_v2_x2_0, "image"),
        "torchvision/mobilenetv2": (torchvision.models.mobilenet_v2, "image"),
        "torchvision/mobilenetv3-small": (torchvision.models.mobilenet_v3_small, "image"),
        "torchvision/mobilenetv3-large": (torchvision.models.mobilenet_v3_large, "image"),
        "torchvision/resnext50": (torchvision.models.resnext50_32x4d, "image"),
        "torchvision/resnext101": (torchvision.models.resnext101_32x8d, "image"),
        "torchvision/wide-resnet50": (torchvision.models.wide_resnet50_2, "image"),
        "torchvision/wide-resnet101": (torchvision.models.wide_resnet101_2, "image"),
        "torchvision/mnasnet0-5": (torchvision.models.mnasnet0_5, "image"),
        "torchvision/mnasnet0-75": (torchvision.models.mnasnet0_75, "image"),
        "torchvision/mnasnet1-0": (torchvision.models.mnasnet1_0, "image"),
        "torchvision/mnasnet1-3": (torchvision.models.mnasnet1_3, "image"),
        "torchvision/fasterrcnn_mobilenet_v3_large_fpn": (torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn, "image"),
        "torchvision/fasterrcnn_mobilenet_v3_large_320_fpn": (torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn, "image"),
        "torchvision/fasterrcnn_resnet50_fpn": (torchvision.models.detection.fasterrcnn_resnet50_fpn, "image"),
        "torchvision/maskrcnn_resnet50_fpn": (torchvision.models.detection.maskrcnn_resnet50_fpn, "image"),
        "torchvision/keypointrcnn_resnet50_fpn": (torchvision.models.detection.keypointrcnn_resnet50_fpn, "image"),
        "torchvision/retinanet_resnet50_fpn": (torchvision.models.detection.retinanet_resnet50_fpn, "image"),
        "torchvision/mc3_18": (torchvision.models.video.mc3_18, "image"),
        "torchvision/r3d_18": (torchvision.models.video.r3d_18, "image"),
        "torchvision/r2plus1d_18": (torchvision.models.video.r2plus1d_18, "image"),
        "torchvision/deeplabv3_mobilenet_v3_large": (torchvision.models.segmentation.deeplabv3_mobilenet_v3_large, "image"),
        "torchvision/deeplabv3_resnet101": (torchvision.models.segmentation.deeplabv3_resnet101, "image"),
        "torchvision/deeplabv3_resnet50": (torchvision.models.segmentation.deeplabv3_resnet50, "image"),
        "torchvision/fcn_resnet50": (torchvision.models.segmentation.fcn_resnet50, "image"),
        "torchvision/fcn_resnet101": (torchvision.models.segmentation.fcn_resnet101, "image"),
        "torchvision/lraspp_mobilenet_v3_large": (torchvision.models.segmentation.lraspp_mobilenet_v3_large, "image"),
    }


def hf_model_builder(model, **kwargs):
    # get the required keys
    import transformers
    _auto_loaders = {
        x: getattr(transformers, x) for x in dir(transformers)
        if x[:4] == "Auto" and x != "AutoConfig"
    }
    auto_model_type = model.split("/")[-1]
    model, auto_model_type, task = model.split("::")

    assert task in ["generation", "masked_lm"], "For now only the following are supported: `generation`, `masked_lm`"

    # initliase the model and tokenizer object
    tokenizer = transformers.AutoTokenizer.from_pretrained(model, **kwargs)
    if not tokenizer.pad_token_id:
        tokenizer.pad_token = "<|endoftext|>"
    model = _auto_loaders[auto_model_type].from_pretrained(model, **kwargs)
    return model, tokenizer, task

### ----- pretrained models master index
# add code based on conditionals, best way is to only include those that
# have proper model building code like transformers, torchvision, etc.

PRETRAINED_MODELS = {}

if is_available("efficientnet_pytorch"):
    PRETRAINED_MODELS.update(load_efficientnet_pytorch_models())

if is_available("torchvision"):
    PRETRAINED_MODELS.update(load_torchvision_models())

if is_available("transformers"):
    PRETRAINED_MODELS["transformers"] = (hf_model_builder, "transformer")


def get_image_models():
    return {k:v for k,v in list(filter(
        lambda x: x[1] == "image", PRETRAINED_MODELS
    ))}

# ---- load function has to manage everything and return Model object properly initialised

def load(model: str, **loader_kwargs):
    if model.startswith("transformers/"):
        # remove the leading text 'transformers/'
        model, tokenizer, task = hf_model_builder(model[13:], **loader_kwargs)
        out = Model(
            model = model,
            category = "text",
            tokenizer = tokenizer
        )

    else:
        model_meta = PRETRAINED_MODELS.get(model, None)
        if model_meta is None:
            raise IndexError(f"Model: {model} not found in storage!")
        model_fn, model_meta = model_meta
        out = Model(
            model = model_fn(pretrained = True),
            category = "image"
        )
    
    return out

