import os
import inspect
from enum import Enum

class OperatorType(Enum):
  """This Enum does not concern the user, however I am describing it so people can get a feel of the breadth
  of what nbox can do. The purpose of ``Operator`` is to build an abstract representation of any possible compute
  and execute them in any fashion needed to improve the overall performance of any distributed software system.
  Here are the different types of operators:

  #. ``UNSET``: this is the default mode and is like using vanilla python without any nbox features.
  #. ``JOB``: In this case the process is run as a batch process and the I/O of values is done using Relics
  #. ``SERVING``: In this case the process is run as an API proces
  #. ``WRAP_FN``: When we wrap a function as an ``Operator``, by default deployed as a job
  #. ``WRAP_CLS``: When we wrap a class as an ``Operator``, by default deployed as a serving
  """
  UNSET = "unset" # default
  JOB = "job"
  SERVING = "serving"
  WRAP_FN = "fn_wrap"
  WRAP_CLS = "cls_wrap"

  def _valid_deployment_types():
    return (OperatorType.JOB.value, OperatorType.SERVING.value)


class _UnsetSpec:
  def __init__(self):
    self.type = OperatorType.UNSET.value

  def __repr__(self):
    return f"[{self.type}]"

class _JobSpec:
  def __init__(
    self,
    job_id,
    rpc_fn_name,
    job,
    workspace_id,
  ):
    from nbox import Job

    self.type = OperatorType.JOB.value
    self.job_id = job_id
    self.rpc_fn_name = rpc_fn_name
    self.job: Job = job
    self.workspace_id = workspace_id

  def __repr__(self):
    return f"[{self.type} {self.job_id} {self.workspace_id}]"

class _ServingSpec:
  def __init__(
    self,
    serving_id,
    rpc_fn_name,
    fn_spec,
    workspace_id,
    track_io: bool = False
  ):
    self.type = OperatorType.SERVING.value
    self.serving_id = serving_id
    self.rpc_fn_name = rpc_fn_name
    self.fn_spec = fn_spec
    self.workspace_id = workspace_id
    self.track_io = track_io

  def __repr__(self):
    return f"[{self.type} {self.serving_id} {self.workspace_id}]"

class _WrapFnSpec:
  def __init__(self, fn_name, wrap_obj):
    self.type = OperatorType.WRAP_FN.value
    self.fn_name = fn_name
    self.wrap_obj = wrap_obj

  def ___repr__(self):
    return f"[{self.type} {self.fn_name}]"

class _WrapClsSpec:
  def __init__(self, cls_name, wrap_obj, init_ak):
    self.type = OperatorType.WRAP_CLS.value
    self.cls_name = cls_name
    self.wrap_obj = wrap_obj
    self.init_ak = init_ak

  def ___repr__(self):
    return f"[{self.type} {self.cls_name}]"


def get_operator_location(_op):
  # get the filepath and name to import for convience
  if _op._op_type == OperatorType.UNSET:
    fp = inspect.getfile(_op.__class__)
    name = _op.__class__.__qualname__
  elif _op._op_type in [OperatorType.JOB, OperatorType.SERVING]:
    raise ValueError("Cannot deploy an operator that is already deployed")
  elif _op._op_type == OperatorType.WRAP_FN:
    fp = _op.__file__
    name = _op.__qualname__[3:] # to account for "fn_"
  elif _op._op_type == OperatorType.WRAP_CLS:
    fp = _op.__file__
    name = _op.__qualname__[4:] # to account for "cls_"
  fp = os.path.abspath(fp) # get the abspath, will be super useful later
  folder, file = os.path.split(fp)
  return fp, folder, file, name