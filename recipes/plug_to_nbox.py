from nbox import plug, PRETRAINED_MODELS

import nbox
import torch
import numpy as np


class DoubleInSingleOut(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.f1 = torch.nn.Linear(2, 4)
        self.f2 = torch.nn.Linear(2, 4)
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, x, y):
        out = self.f1(x) + self.f2(y)
        logit_scale = self.logit_scale.exp()
        out = logit_scale - out @ out.t()
        return out


def my_model_builder_fn(**kwargs):
    # let it accept **kwargs, you use what you need
    # each method must return two things the model itself, and some extra args
    return DoubleInSingleOut(), {}


num_sources = len(PRETRAINED_MODELS)
print("           number of models in the registry:", num_sources)

# plug the model
plug(
    "my_model_method",  # what should be the name / key
    my_model_builder_fn,  # method that will be called to build the model
    {"x": "image", "y": "image"},  # what are the categories of inputs and outputs
)
new_num_sources = len(PRETRAINED_MODELS)
print("number of models in the registry after plug:", new_num_sources)

# loading my mew model
model = nbox.load("my_model_method")

out = model({"x": torch.randn(4, 2).numpy(), "y": torch.randn(4, 2).numpy()})

print(out.shape)
