from functools import partial
from tempfile import gettempdir
from typing import Dict

from nbox import Model
import nbox.utils as U
from nbox.framework import NboxOptions
from nbox.framework.ml import TorchToTorchscript
from nbox.framework.model_spec_pb2 import ModelSpec

import numpy as np
import torch

def pre_fn(x) -> Dict:
  from torchvision.transforms import functional as trfn
  from nbox import utils as U
  
  if isinstance(x, str):
    x = U.get_image(x)
    x = trfn.to_tensor(x)
    x = trfn.resize(x, (44, 44))
  x = trfn.normalize(x, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
  return {"x": x}

def post_fn(pred):
  out = pred.argmax(dim=-1)
  return out.tolist()

class Feedforward(torch.nn.Module):
  def __init__(self, input_size, hidden_size):
    super(Feedforward, self).__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.network = torch.nn.Sequential(
      torch.nn.Linear(self.input_size, self.hidden_size),
      torch.nn.Softmax(-1),
      torch.nn.Linear(self.hidden_size, 1),
      torch.nn.Sigmoid(),
    )

  def forward(self, x):
    output = self.network(x)
    return output

def test_feedforward():
  x = np.random.uniform(size=(1, 3, 44, 44))
  x = torch.tensor(x).float()
  model = Feedforward(44, 2)

  src_model = Model(model, method = None, pre = pre_fn, post = post_fn)

  print("Source output:", np.array(src_model(x)).shape)

  # avoid numpy problems
  with torch.no_grad():
    model_spec: ModelSpec = src_model.torch_to_torchscript(
      # get the dataclass just for your stub function, provide precise inputs
      # for your export method
      TorchToTorchscript(func = src_model.model, example_inputs = x,),

      # create a folder /tmp/nbox_testing
      NboxOptions(model_name = None, folder = U.join(gettempdir(), "nbox_testing"), create_folder = True)
    )
  print(model_spec)

  model_spec.requirements.extend(["torchvision"]) # add the dependencies for pre_fn

  # model_spec, nbx_path = src_model.deploy(
  #   model_spec,
  #   deployment_id_or_name="ywrrpw8g",
  #   workspace_id="zcxdpqlk",
  #   _unittest = True
  # )
  # print(model_spec)

  # trg_model = Model.deserialise(model_spec, nbx_path)
  # print(trg_model)
  # print("trg_out:", np.array(trg_model(x)).shape)

  url, key = src_model.deploy(
    model_spec,
    deployment_id_or_name="ywrrpw8g",
    workspace_id="zcxdpqlk",
    _unittest = False
  )
  print(url, key)


test_feedforward()


# def test_hf():
#   def pre_(x):
#     import transformers
#     token = transformers.AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
#     return token(x, return_tensors = "pt")

#   def post_(x):
#     values, indices = x.last_hidden_state[0, 0].topk(10)
#     values = values.tolist()
#     indices = indices.tolist()
#     return {
#       "result": [{i:v} for i, v in zip(indices, values)]
#     }
  
#   import transformers

#   m = Model(
#     transformers.AutoModel.from_pretrained("prajjwal1/bert-tiny"), pre_, post_
#   )
#   first_output = m("yash bonde")
#   print(first_output)

#   export_fn = m.create_export_fn("torchscript", "./filepath.pt")
#   export_fn = partial(export_fn, ...) # user loads a partial

#   new_model = Model.deserialise(
#     m.serialise(
#       export_fn = export_fn,
#       model_name = "test69",
#       export_type = "torchscript",
#       _unit_test = False
#     )
#   )
#   second_output = new_model("yash bonde")
#   print(second_output)

# test_hf()


# def test_resnet_torchscript():
#   x = randn(1, 3, 44, 44)
#   # load resnet model with preprocessing function
#   resnet = load(
#   "torchvision/resnet18",
#   pre_fn,
#   )
#   #Model(i0: Tensorflow.Object, i1: )
#   resnet.eval()
#   first_out = resnet(x).outputs

#   # serialise then deserialise
#   new_resnet = Model.deserialise(
#   resnet.serialise(
#     input_object = x,
#     model_name = "test69",
#     export_type = "torchscript",
#     _unit_test = False
#     )
#   )

#   # now pass data through the new model
#   second_out = new_resnet(x).outputs
#   assert torch.equal(first_out, second_out)
#   return second_out.topk(10)

# def test_resnet_onnx():
#   x = randn(1, 3, 44, 44)
#   # load resnet model with preprocessing function
#   resnet = load(
#   "torchvision/resnet18",
#   pre_fn,
#   )
#   #Model(i0: Tensorflow.Object, i1: )
#   resnet.eval()
#   first_out = resnet(x).outputs

#   # serialise then deserialise
#   new_resnet = Model.deserialise(
#   resnet.serialise(
#     input_object = x,
#     model_name = "test69",
#     export_type = "onnx",
#     _unit_test = False
#     )
#   )

#   # now pass data through the new model
#   second_out = new_resnet(x).outputs[0][0]
#   first_out = first_out.squeeze().cpu().detach().numpy()
#   print("second out",second_out.shape)
#   print("first out",first_out.shape)
#   print("difference", np.subtract(first_out,second_out))
#   assert np.allclose(first_out, second_out, atol=1E-5)
#   return torch.tensor(second_out).topk(10)


def br():
  print("\n")
  print("#"*50, "\n")

# #Test Feedforward -
# br()

# #Test Resnet from Torchscript-
# br()
# print("Resnet from Torchscript: \n", test_resnet_torchscript())

#Test Resnet from ONNX-
# br()
# print("Resnet from ONNX: \n", test_resnet_onnx())
