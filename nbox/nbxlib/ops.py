"""MIT-License code for all operators that are open sourced"""

import shlex
import subprocess

from ..operator import Operator
from ..utils import PoolBranch
from ..instance import Instance

# nbox/

class YATW(Operator):
  # yet another time waster
  def __init__(self, s=3):
    super().__init__()
    self.s = s

  def forward(self, i = ""):
    import random
    from time import sleep

    y = self.s + random.random()
    print(f"Sleeping (YATW-{i}): {y}")
    sleep(y)


def init_function(s=0.1):
  # this random fuction sleeps for a while and then returns number
  from random import randint
  from time import sleep

  y = randint(4, 8) + s
  print(f"Sleeping (fn): {y}")
  sleep(y)
  return y


class Magic(Operator):
  # named after the Dr.Yellow train that maintains the Shinkansen
  def __init__(self):
    super().__init__()
    self.init = Python(init_function) # waste time and return something
    self.foot = YATW() # Oh, my what will happen here?
    self.cond1 = ShellCommand("echo 'Executing condition {cond}'")
    self.cond2 = ShellCommand("echo 'This is the {cond} option'")
    self.cond3 = ShellCommand("echo '{cond} times the charm :P'")

  def forward(self):
    t1 = self.init()
    self.foot(0)

    if t1 > 6:
      self.cond1(cond = 1)
    elif t1 > 10:
      self.cond2(cond = "second")
    else:
      self.cond3(cond = "Third")


class NboxInstanceStartOperator(Operator):
  def __init__(self, instances):
    """Starts multiple instances on nbox in a blocking fashion"""
    super().__init__()
    if not isinstance(instances, list):
      instances = [instances]
    assert instances[0].__class__ == Instance, "instances must be of type Instance"
    self.instances = instances
    self.pool = PoolBranch("thread", len(instances), _name = "instance_starter")

  def forward(self):
    self.pool(
      lambda instance: instance.start(cpu_only = True),
      self.instances
    )
    return None

class NboxModelDeployOperator(Operator):
  def __init__(self, model_name, model_path, model_weights, model_labels):
    """Simple Operator that wraps the deploy function of ``Model``"""
    super().__init__()
    self.model_name = model_name
    self.model_path = model_path
    self.model_weights = model_weights
    self.model_labels = model_labels

  def forward(self, name):
    from ..model import Model
    Model(
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

# /nbox

# arch/

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

# /arch

class Python(Operator):
  def __init__(self, func, *args, **kwargs):
    """Convert a python function into an operator, everything has to be passed at runtime"""
    super().__init__()
    self.fak = (func, args, kwargs)

  def forward(self):
    return self.fak[0](*self.fak[1], **self.fak[2])

class GitClone(Operator):
  def __init__(self, url, path = None, branch = None):
    """Clone a git repository into a path"""
    super().__init__()
    self.url = url
    self.path = path
    self.branch = branch

  def forward(self):
    # git clone -b <branch> <repo> <path>
    import subprocess
    command = ["git", "clone"]
    if self.branch:
      command.append("-b")
      command.append(self.branch)
    if self.path:
      command.append(self.path)
    command.append(self.url)
    subprocess.run(command, check = True)

class ShellCommand(Operator):
  def __init__(self, *commands):
    """Run multiple shell commands, uses ``shelex`` to prevent injection"""
    super().__init__()
    import string

    self.commands = commands
    all_in = []
    for c in self.commands:
      all_in.extend([tup[1] for tup in string.Formatter().parse(c) if tup[1] is not None])
    self._inputs = all_in

  def forward(self, *args, **kwargs):
    for comm in self.commands:
      comm = comm.format(*args, **kwargs)
      comm = shlex.split(comm)
      subprocess.run(comm, check = True)

class Notify(Operator):
  _mode_to_packages = {
    "slack": "slackclient",
    "ms_teams": "microsoft-teams",
    "discord": "discord",
  }

  def __init__(
    self,
    slack_connect: str = None,
    ms_teams: str = None,
    discord: str = None,
  ):
    """Notifications"""
    super().__init__()
    self.notify_mode = None
    self.notify_id = None

    for mode, id in [
      ("slack", slack_connect),
      ("ms_teams", ms_teams),
      ("discord", discord),
    ]:
      if id:
        self.notify_mode = mode
        self.notify_id = id

        # check for package dependencies
        import importlib
        try:
          importlib.import_module(self._mode_to_packages[mode])
        except ImportError:
          raise Exception(f"{self._mode_to_packages[mode]} package required for {mode}")
        break
    
    if not self.notify_mode:
      raise Exception("No notification mode specified")

  def forward(self, message: str, **kwargs):
    package = self._mode_to_packages[self.notify_mode]
    import importlib
    importlib.import_module(package)
    if self.notify_mode == "slack":
      from slackclient import SlackClient
      sc = SlackClient(self.notify_id)
      sc.api_call("chat.postMessage", text = message, **kwargs)
    elif self.notify_mode == "ms_teams":
      from microsoft_teams.api_client import TeamsApiClient
      from microsoft_teams.models import MessageCard
      client = TeamsApiClient(self.notify_id)
      client.connect()
      client.send_message(MessageCard(text = message, **kwargs))
    elif self.notify_mode == "discord":
      from discord import Webhook, RequestsWebhookAdapter
      webhook = Webhook.from_url(self.notify_id, adapter = RequestsWebhookAdapter())
      webhook.send(message, **kwargs)

