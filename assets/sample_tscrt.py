# simple torch neural network and export as torchscript

import nbox

import torch
from torch import nn

# class Model(nn.Module):
#   def __init__(self):
#     super().__init__()

#     self.h = nn.Sequential(
#       nn.Linear(2, 3),
#       nn.ReLU(),
#       nn.Linear(3, 1),
#       nn.Sigmoid()
#     )

#   def forward(self, x):
#     return self.h(x)

# model = Model()
# _in = torch.randn(5, 2)
# inputs = [_in]
# print(_in.shape)
# out = model(_in)
# print(out.shape)

# traced_model = torch.jit.trace(model, _in)
# print(traced_model)

# print("::" * 10)
# out = traced_model(_in)
# print(out.shape)
# outputs = [out]

model = nbox.load("torchvision/mobilenetv2", pretrained=True)
out, in_ = model([torch.randn(400, 400, 3) for _ in range(2)], return_inputs=True)
inputs = [in_]
outputs = [out]
traced_model = torch.jit.trace(model.model, in_, check_tolerance=0.0001)

torch.jit.save(traced_model, "./sample.pt")

# create the metadata for the model that stores input and output shapes
# and types

meta = {
    "metadata": {
        "inputs": {
            f"input_{i}": {
                "dtype": str(x.dtype),
                "tensorShape": {"dim": [{"name": "", "size": y} for y in x.shape], "unknownRank": False},
                "name": f"input_{i}",
            }
            for i, x in enumerate(inputs)
        },
        "outputs": {
            f"output_{i}": {
                "dtype": str(x.dtype),
                "tensorShape": {"dim": [{"name": "", "size": y} for y in x.shape], "unknownRank": False},
                "name": f"output_{i}",
            }
            for i, x in enumerate(outputs)
        },
    },
    "spec": {"name": "some_model_name", "category": model.category, "model_key": model.model_key},
}

import json

with open("./sample.json", "w") as f:
    f.write(json.dumps(meta, indent=2))

# from pprint import pprint as peepee

# peepee(meta)

# import requests
# print("::" * 10)
# r = requests.get(
#   "https://api.test-2.nimblebox.ai/yash_bonde_139/slow_steel_bfa8/metadata",
#   headers = {
#     "NBX-KEY": "nbxdeploy_ocU8op7ZOU86MOXHceldGNLz20DJALkGYRc4od7t"
#   }
# )
# peepee(r.json()["metadata"]["signature_def"]["signatureDef"]["serving_default"])
