# MIT-License code for all operators that are open sourced

from .operator import Operator

from ..utils import Pool
from ..jobs import Instance

# nbox/

class NboxInstanceStartOperator(Operator):
  def __init__(self, instances):
    super().__init__()
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


class WaitTillJIDComplete(Operator):
  def __init__(self, instance, jid):
    super().__init__()
    self.instance = instance
    self.jid = jid

  def forward(self, poll_interval = 5):
    status = self.instance(self.jid)
    if status == "done":
      return None
    elif status == "error-done":
      raise Exception("Job {} failed".format(self.jid))
    elif status == "running":
      from time import sleep
      while status == "running":
        sleep(poll_interval)
        status = self.instance(self.jid)
      if status == "done":
        return None
      elif status == "error-done":
        raise Exception("Job {} failed".format(self.jid))

# /nbox
