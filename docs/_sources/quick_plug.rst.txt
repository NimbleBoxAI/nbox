
How to plug your own models
===========================

File: `plug_to_nbox.py <https://github.com/NimbleBoxAI/nbox/blob/staging/recipes/plug_to_nbox.py>`_

With ``nbox`` you can load many public models or models on NBX Deploy. If you are using ``nbox`` in your own codebase then you would like to have a way to plug your models at one place and keep the API for loading consistent, this is done by using ``nbox.plug`` method.

First step is to define any model:

.. code-block:: python

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

And then define a builder function:

.. code-block:: python

   def my_model_builder_fn(**kwargs):
       # let it accept **kwargs, you use what you need
       # each method must return two things the model itself, and some extra args
       return DoubleInSingleOut(), {}

Then all you need to do is ``plug`` the model:

.. code-block:: python

   # plug the model
   plug(
       "my_model_method",             # what should be the name / key
       my_model_builder_fn,           # method that will be called to build the model
       {"x": "image", "y": "image"},  # what are the categories of inputs and outputs
   )

And then start using it else where:

.. code-block:: python

   model = nbox.load("my_model_method") # loading my mew model
   out = model({"x": torch.randn(4, 2).numpy(), "y": torch.randn(4, 2).numpy()})
   print(out.shape)
