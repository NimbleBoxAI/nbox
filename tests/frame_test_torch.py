from nbox import load, Model
from nbox.utils import get_image
import numpy as np
from torch import randn
import torch

def pre_fn(x):
  from torchvision.transforms import functional as trfn

  if isinstance(x, str):
    x = get_image(x)
    x = trfn.to_tensor(x)
  x = trfn.normalize(x, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
  return {
    "x": x
  }


def test_feedforward():
    class Feedforward(torch.nn.Module):
        def __init__(self, input_size, hidden_size):
            super(Feedforward, self).__init__()
            self.input_size = input_size
            self.hidden_size  = hidden_size
            self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
            self.soft1 = torch.nn.Softplus()
            self.fc2 = torch.nn.Linear(self.hidden_size, 1)
            self.sigmoid = torch.nn.Sigmoid()
        def forward(self, x):
            hidden = self.fc1(x)
            soft = self.soft1(hidden)
            output = self.fc2(soft)
            output = self.sigmoid(output)
            return output

    x = np.random.uniform(size=(1, 224))
    x = torch.tensor(x).float()
    model = Feedforward(224, 2)
    model = Model(model, None)
    model.eval()
    first_out = model(x).outputs
    new_model = Model.deserialise(
    model.serialise(
        input_object = x,
        model_name = "test69",
        export_type = "torchscript",
        _unit_test = False
        )
    )
    second_out = new_model(x).outputs
    assert torch.equal(first_out, second_out)
    return second_out



def test_resnet_torchscript():
    x = randn(1, 3, 44, 44)
    # load resnet model with preprocessing function
    resnet = load(
    "torchvision/resnet18",
    pre_fn,
    )
    #Model(i0: Tensorflow.Object, i1: )
    resnet.eval()
    first_out = resnet(x).outputs

    # serialise then deserialise
    new_resnet = Model.deserialise(
    resnet.serialise(
        input_object = x,
        model_name = "test69",
        export_type = "torchscript",
        _unit_test = False
        )
    )

    # now pass data through the new model
    second_out = new_resnet(x).outputs
    assert torch.equal(first_out, second_out)
    return second_out.topk(10)

def test_resnet_onnx():
    x = randn(1, 3, 44, 44)
    # load resnet model with preprocessing function
    resnet = load(
    "torchvision/resnet18",
    pre_fn,
    )
    #Model(i0: Tensorflow.Object, i1: )
    resnet.eval()
    first_out = resnet(x).outputs

    # serialise then deserialise
    new_resnet = Model.deserialise(
    resnet.serialise(
        input_object = x,
        model_name = "test69",
        export_type = "onnx",
        _unit_test = False
        )
    )

    # now pass data through the new model
    second_out = new_resnet(x).outputs[0][0]
    first_out = first_out.squeeze().cpu().detach().numpy()
    print("second out",second_out)
    print("first out",first_out)
    assert np.array_equal(first_out, second_out)
    return second_out.topk(10)


def br():
    print("\n")
    print("#"*50, "\n")

#Test Feedforward -
br()
print("FeedForward: \n", test_feedforward())

#Test Resnet from Torchscript-
br()
print("Resnet from Torchscript: \n", test_resnet_torchscript())

#Test Resnet from ONNX-
br()
print("Resnet from ONNX: \n", test_resnet_onnx())
