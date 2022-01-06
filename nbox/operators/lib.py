# MIT-License code for all operators that are open sourced

from .operator import Operator

from ..utils import Pool
from ..jobs import Instance

# nbox/

class NboxInstanceStartOperator(Operator):
  def __init__(self, instances):
    super().__init__("instance_starter")
    if not isinstance(instances, list):
      instances = [instances]
    assert instances[0].__class__ == Instance, "instances must be of type Instance"
    self.instances = instances
    self.pool = Pool("thread", len(instances), _name = "instance_starter")

  def forward(self):
    self.pool(
      lambda instance: instance.start(cpu_only = True),
      self.instances
    )
    return None

# /nbox
