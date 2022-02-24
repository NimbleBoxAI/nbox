from typing import Sequence
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from nbox.model import Model

class MLP(nn.Module):
  features: Sequence[int]

  @nn.compact
  def __call__(self, x):
    for feat in self.features[:-1]:
      x = nn.relu(nn.Dense(feat)(x))
    x = nn.Dense(self.features[-1])(x)
    x = nn.Dense(features=2)(x)
    x = nn.log_softmax(x)
    return x

model = MLP([12, 8, 4])
batch = np.random.rand(2, 10)
variables = model.init(jax.random.PRNGKey(0), batch)

m = Model(model, variables)
print(m(batch).outputs)