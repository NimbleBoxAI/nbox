from nbox import Operator
from nbox.instance import Instance
from nbox.utils import PoolBranch

class NboxInstanceStartOperator(Operator):
  def __init__(self, instances):
    """Starts multiple instances on nbox in a blocking fashion"""
    super().__init__()
    if not isinstance(instances, list):
      instances = [instances]
    assert instances[0].__class__ == Instance, "instances must be of type nbox.Instance"
    self.instances = instances
    self.pool = PoolBranch("thread", len(instances), _name = "instance_starter")

  def forward(self):
    self.pool(
      lambda instance: instance.start(cpu_only = True),
      self.instances
    )

class NboxModelDeployOperator(Operator):
  def __init__(self, model_name, model_path, model_weights, model_labels):
    """Simple Operator that wraps the deploy function of ``Model``"""
    super().__init__()
    self.model_name = model_name
    self.model_path = model_path
    self.model_weights = model_weights
    self.model_labels = model_labels

  def forward(self, name):
    from nbox.model import Model

    return Model(
      self.model_name,
      self.model_path,
      self.model_weights,
      self.model_labels,
    ).deploy(name)

class NboxWaitTillJIDComplete(Operator):
  def __init__(self, instance, jid):
    """Blocks threads while a certain PID is complete on Instance"""
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

class NboxInstanceMv(Operator):
  def __init__(self, i: str, workspace_id: str) -> None:
    super().__init__()
    self.build = Instance(i = i, workspace_id = workspace_id)
    self.start_build = NboxInstanceStartOperator(self.build)
  
  def forward(self, src: str, dst: str, force: bool = False) -> None:
    if not self.build.is_running():
      self.start_build()
    resp = self.build.mv(src, dst, force = force)
    return resp
