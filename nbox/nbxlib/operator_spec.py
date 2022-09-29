import os
import inspect
from enum import Enum
from threading import Thread, Lock
from time import sleep

import nbox.utils as U
from nbox.utils import logger
from nbox.hyperloop.job_pb2 import Resource

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


class _LocalMapFuture:
  def __init__(self, result_folder, tag):
    self.fp = U.join(result_folder, tag + "_output")

  def __repr__(self) -> str:
    return f"_LocalMapFuture ({self.fp})"

  def wait(self):
    while not self.done():
      sleep(0.1)

  def done(self):
    os.path.exists(self.fp)

  def result(self):
    if not self.done():
      logger.error("Cannot get result before it is done")
      return None
    return U.from_pickle(self.fp)


class _LocalMapPooler:
  def __init__(self, folder, file, name, n_workers = 10):
    self.folder = folder
    self.file = file
    self.name = name
    self.n_workers = n_workers
    
    # things to manage the processes
    self.items = []
    self.processes = []
    self.running = {}
    self.completed = 0
    self._mutex = Lock()
    self._break = False

    self.event_loop = Thread(target = self._event_loop, daemon = True)

  def __repr__(self):
    return f"_LocalMapPooler ({self.n_workers:03d} workers) running '{self.folder}/{self.file}::{self.name}'"

  def _there_is_space(self):
    return len(self.processes) < self.n_workers

  def _event_loop(self):
    # this sees what is in the queue if any slot is finished and runs it
    while True:
      with self._mutex:
        n_items = len(self.items)
        _break = self._break
      print(_break, n_items)
      if _break:
        return
      if not n_items:
        sleep(0.1)

      status = [p.poll() for p in self.processes]
      success = [x == 0 for x in status]
      errored = [x is not None and x != 0 for x in status]
      completed = sum([s is not None for s in status])
      logger.debug(f"done: [{completed}/{len(status)}], success: {sum(success)}, errored: {sum(errored)}")

      pending = n_items - completed
      while pending and self._there_is_space():
        with self._mutex:
          try:
            (tag, fn) = self.items.pop(0)
          except IndexError:
            break

        print("Processing tag", tag)
        p = fn()
        self.processes.append(p)
        pending -= 1

  def submit(self, tag, fn, result_folder) -> _LocalMapFuture:
    self.items.append((tag, fn))
    fut = _LocalMapFuture(result_folder, tag)
    return fut

  def start(self):
    self.event_loop.start()

  def stop(self):
    with self._mutex:
      self._break = True
    self.event_loop.join()



DEFAULT_RESOURCE = Resource(
  cpu = "100m",         # 100mCPU
  memory = "512Mi",     # MiB
  disk_size = "5Gi",    # GiB
  gpu = "none",         # keep "none" for no GPU
  gpu_count = "0",      # keep "0" when no GPU
  timeout = 120_000,    # 2 minutes between attempts
  max_retries = 2,      # third times the charm :P
)

# we will keep on expanding this list, note that this cannot be directly used with copytree,
# to make it work remove the trailing slash
FN_IGNORE = [
  "__pycache__/", "venv/", ".git/", ".vscode/"
]
