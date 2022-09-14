"""
# NBX-Model

This will combine all the NimbleBox modules into a single callable.
"""

from nbox.operator import Operator

class Model(Operator):
  """Every computation element in the NBX-StratoMembrane is an ``Operator``, thus a model is also just
  a special type of ``Operator``. User will subclass this and wrap their operation specific code in
  the functions"""
  def __init__(self,):
    """Top of the stack Model class."""
    super().__init__()

  def _metadata(self):
    return {"config": "fill _metadata method"}

  def __repr__(self):
    return f"<nbox.Model: {self.model} >"

  def forward(self):
    raise NotImplementedError("Implement your check back later.")

  def load(self, fpath: str):
    # load the model from fpath
    pass

  def save(self, fpath: str):
    # save the model to the fpath
    pass
