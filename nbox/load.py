import transformers
from efficientnet_pytorch import EfficientNet

from nbox import Model

# --- functions


def load_torchvision_models():
    import torchvision

    # all_ = [x for x in dir(torchvision.models) if re.match(r"[a-z]", x[0])]
    return {
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
    }


### ----- pretrained models master caller

PRETRAINED_MODELS = {
    "image": {
        "efficientnet_from_name": EfficientNet.from_name,
        "efficientnet_pretrained": EfficientNet.from_pretrained,
    },
    "video": {},
    "transformers": {
        "generation": transformers.AutoModelForCausalLM,
    },
}

# load models based on some conditionals like available packages, versions, etc.
# check out how transformers does not require any package to instantiate
PRETRAINED_MODELS["image"].update(load_torchvision_models())

# ----


def load(model: str, pretrained: bool = False, **kwargs):
    if model[:12] == "efficientnet":
        if pretrained:
            model_builder = PRETRAINED_MODELS["image"].get("efficientnet_from_pretrained")
            return Model(model_builder(model, **kwargs), "image")

        model_builder = PRETRAINED_MODELS["image"].get("efficientnet_from_name")
        return Model(model_builder(model, **kwargs), "image")

    else:
        for data_type in PRETRAINED_MODELS.keys():
            # Picks the first ket incase of two duplicates
            model_builder = PRETRAINED_MODELS[data_type].get(model, None)
            if model_builder != None:
                break
        if model_builder == None:
            raise KeyError("Model not found in the list")
        return Model(model_builder(pretrained, **kwargs), data_type)
