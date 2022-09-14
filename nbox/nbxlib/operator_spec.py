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

class _JobSpec:
  def __init__(
    self,
    job_id,
    rpc_fn_name,
    job,
    workspace_id,
  ):
    self.type = OperatorType.JOB.value
    self.job_id = job_id
    self.rpc_fn_name = rpc_fn_name
    self.job = job
    self.workspace_id = workspace_id

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

class _WrapFnSpec:
  def __init__(self, fn_name, wrap_obj):
    self.type = OperatorType.WRAP_FN.value
    self.fn_name = fn_name
    self.wrap_obj = wrap_obj

class _WrapClsSpec:
  def __init__(self, cls_name, wrap_obj, init_ak):
    self.type = OperatorType.WRAP_CLS.value
    self.cls_name = cls_name
    self.wrap_obj = wrap_obj
    self.init_ak = init_ak
