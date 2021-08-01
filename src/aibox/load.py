import torch
import numpy
from PIL import Image
from torchvision import models
from efficientnet_pytorch import EfficientNet

PRETRAINED_MODELS = {
"image": {
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
  },
"video": {
  
  }
}

class Processing:
  # Define processing methods for different input types, eg video, images,
  # text, etc... so they can be called later in the Models class in the
  # __call__ method.

  # Probably also add something like transformation from torchvision.transforms
  @staticmethod
  def image_processing(input_path):
    img = Image.open(input_path).convert('RGB')
    img = torch.tensor(numpy.asarray(img))
    img = img.permute(2, 0, 1).unsqueeze(0)
    return img.float()
 
class Model:

  def __init__(self, model: torch.nn.Module, dtype: str = None):
    self.model = model
    self.dtype = dtype

  def data_type(self):
    return self.dtype

  def available_dtypes(self):
    """
    Returns a dict of available data types as keys and the respective
    processing method as value.
    """
    return {'image': Processing.image_processing}

  def get_model(self):
    return self.model

  def __call__(self, input_path):
    # Somehow determine input_type as image, video, text, etc...
    # Kept as image for testing.
    input_type = "image"
    assert input_type == self.dtype, "Given {} file, this model only supports {}" \
      "files.".format(input_type, dtype)
    input_processor = self.available_dtypes().get(self.dtype, None)
    input_tensor = input_processor(input_path)
    return self.model(input_tensor)


def load(model: str, pretrained: bool = False, **kwargs):
  if model.split("-")[0] == "efficientnet":
    if pretrained:
      model_builder = PRETRAINED_MODELS["image"].get(
          "efficientnet_from_pretrained")
      return Model(model_builder(model, **kwargs), "image")

    model_builder = PRETRAINED_MODELS["image"].get("efficientnet_from_name")
    return Model(model_builder(model, **kwargs), "image")

  else:
    for data_type in PRETRAINED_MODELS.keys():
      # Assumes there are no two same keys even in two different dtype dicts.
      model_builder = PRETRAINED_MODELS[data_type].get(model, None)
      if model_builder != None:
        break
    if model_builder == None:
      raise KeyError("Model not found in the list")
    return Model(model_builder(pretrained, **kwargs), data_type)
