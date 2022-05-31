from typing import Dict, List
from nbox import Operator

class StepOp(Operator):
  def __init__(self):
    """Convinience operator, add a no output operator using ``.add_step`` and don't write forward

    Usage
    -----

    .. code-block:: python

    class InstallPython(StepOp)
      def __init__(self, version: str = "3.9):
        self.add_step(ShellCommand(f"chmod +x ./scripts/python{version}_install.sh"))
        self.add_step(ShellCommand(f"./scripts/python{version}_install.sh"))

    install_python = InstallPython() # init op
    install_python() # call it without defining the forward function
    """
    super().__init__()
    self.steps = []

  def add_step(self, step: Operator):
    self.steps.append(step)

  def forward(self):
    for step in self.steps:
      step()

class Sequential():
  def __init__(self, *ops):
    """Package a list of operators into a sequential pipeline"""
    super().__init__()
    for op in ops:
      assert isinstance(op, Operator), "Operator must be of type Operator"
    self.ops = ops

  def forward(self, x = None, capture_output = False):
    out = x
    outputs = []
    for op in self.ops:
      out = op(out)
      outputs.append(out)
    if capture_output:
      return out, outputs
    return out
