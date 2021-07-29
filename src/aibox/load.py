from torchvision import models
from efficientnet_pytorch import EfficientNet

PRETRAINED_MODELS = {
  "efficientnet_from_name": EfficientNet.from_name,
  "efficientnet_pretrained": EfficientNet.from_pretrained,
  "alexnet": models.alexnet,
  "resnet18": models.resnet18,
  "resnet34": models.resnet34,
  "resnet50": models.resnet50,
  "resnet101": models.resnet101,
  "resnet152": models.resnet152,
  "vgg11": models.vgg11,
  "vgg11-bn": models.vgg11_bn,
  "vgg13": models.vgg13,
  "vgg13-bn": models.vgg13_bn,
  "vgg16": models.vgg16,
  "vgg16-bn": models.vgg16_bn,
  "vgg19": models.vgg19,
  "vgg19-bn": models.vgg19_bn,
  "squeezenet1": models.squeezenet1_0,
  "squeezenet1-1": models.squeezenet1_1,
  "densenet121": models.densenet121,
  "densenet161": models.densenet161,
  "densenet169": models.densenet169,
  "densenet201": models.densenet201,
  "inceptionv3": models.inception_v3,
  "googlenet": models.googlenet,
  "shufflenetv2-0.5": models.shufflenet_v2_x0_5,
  "shufflenetv2-1.0": models.shufflenet_v2_x1_0,
  "shufflenetv2-1.5": models.shufflenet_v2_x1_5,
  "shufflenetv2-2.0": models.shufflenet_v2_x2_0,
  "mobilenetv2": models.mobilenet_v2,
  "mobilenetv3-small": models.mobilenet_v3_small,
  "mobilenetv3-large": models.mobilenet_v3_large,
  "resnext50": models.resnext50_32x4d,
  "resnext101": models.resnext101_32x8d,
  "wide-resnet50": models.wide_resnet50_2,
  "wide-resnet101": models.wide_resnet101_2,
  "mnasnet0-5": models.mnasnet0_5,
  "mnasnet0-75": models.mnasnet0_75,
  "mnasnet1-0": models.mnasnet1_0,
  "mnasnet1-3": models.mnasnet1_3
}

def load(model: str, pretrained: bool = False, **kwargs):
  if model.split("-")[0] == "efficientnet":
    if pretrained:
      model_builder = PRETRAINED_MODELS.get("efficientnet_from_pretrained")
      return model_builder(model, **kwargs)

    model_builder = PRETRAINED_MODELS.get("efficientnet_from_name")
    return model_builder(model, **kwargs)

  else:
    model_builder = PRETRAINED_MODELS.get(model, None)
    if model_builder == None:
      raise KeyError("Model not found in the list")

    return model_builder(pretrained, **kwargs)
