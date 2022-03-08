# Some parts of the code are based on the pytorch nn.Module class
# pytorch license: https://github.com/pytorch/pytorch/blob/master/LICENSE
# due to requirements of stability, some type enforcing is performed

import os
import json
import zipfile
from typing import Callable, Union
from functools import partial
from tempfile import gettempdir, mkdtemp
from collections import OrderedDict
from datetime import datetime, timedelta

from ..network import deploy_job, Cron
from ..utils import logger
from .. import utils as U
from ..framework import AirflowMixin, PrefectMixin, LuigiMixin
from ..framework.on_functions import get_nbx_flow, DBase
from ..init import nbox_grpc_stub, nbox_ws_v1
from ..jobs import Job


class StateDictModel(DBase):
  __slots__ = [
    "state", # :str
    "data", # :dict
    "inputs", # :dict
    "outputs", # :dict
  ]


class Tracer:
  def __init__(self, tracer = None):
    if tracer == "stub":
      # when job is running on NBX, gRPC stubs are used
      if nbox_grpc_stub == None:
        raise RuntimeError("nbox_grpc_stub is not initialized")
    self.tracer = tracer

  def __call__(self, dag_update):
    if self._trace_obj == "stub":
      from grpc import RpcError
      from ..hyperloop.nbox_ws_pb2 import UpdateRunRequest
      from ..hyperloop.job_pb2 import NBXAuthInfo

      dag = dag_update["dag"]

      try:
        response = nbox_grpc_stub.UpdateRun(
          UpdateRunRequest(job=Job(id="jt3earah", dag=None, status="COMPLETED", auth_info=NBXAuthInfo(workspace_id="zcxdpqlk")))
        )
      except RpcError as e:
        logger.error(f"Could not update job {self.id}")
        raise e
    else:
      logger.info(dag_update)


class Operator(AirflowMixin, PrefectMixin, LuigiMixin):
  _version: int = 1 # always try to keep this an i32

  def __init__(self) -> None:
    self._operators = OrderedDict() # {name: operator}
    self._op_trace = []
    self._trace_object = Tracer()

  # mixin/

  # AirflowMixin methods
  # --------------------
  # AirflowMixin.to_airflow_operator(self, timeout, **operator_kwargs):
  # AirflowMixin.to_airflow_dag(self, dag_kwargs, operator_kwargs)
  # AirflowMixin.from_airflow_operator(cls, air_operator)
  # AirflowMixin.from_airflow_dag(cls, dag)

  # PrefectMixin methods
  # --------------------
  # PrefectMixin.from_prefect_flow()
  # PrefectMixin.to_prefect_flow()
  # PrefectMixin.from_prefect_task()
  # PrefectMixin.to_prefect_task()
  
  # LuigiMixin methods
  # -----------------
  # LuigiMixin.from_luigi_flow()
  # LuigiMixin.to_luigi_flow()
  # LuigiMixin.from_luigi_task()
  # LuigiMixin.to_luigi_task()

  # /mixin

  def __repr__(self):
    # from torch.nn.Module
    def _addindent(s_, numSpaces):
      s = s_.split('\n')
      # don't do anything for single-line stuff
      if len(s) == 1:
        return s_
      first = s.pop(0)
      s = [(numSpaces * ' ') + line for line in s]
      s = '\n'.join(s)
      s = first + '\n' + s
      return s

    # We treat the extra repr like the sub-module, one item per line
    extra_lines = []
    extra_repr = ""
    # empty string will be split into list ['']
    if extra_repr:
      extra_lines = extra_repr.split('\n')
    child_lines = []
    for key, module in self._operators.items():
      mod_str = repr(module)
      mod_str = _addindent(mod_str, 2)
      child_lines.append('(' + key + '): ' + mod_str)
    lines = extra_lines + child_lines

    main_str = self.__class__.__name__ + '('
    if lines:
      # simple one-liner info, which most builtin Modules will use
      if len(extra_lines) == 1 and not child_lines:
        main_str += extra_lines[0]
      else:
        main_str += '\n  ' + '\n  '.join(lines) + '\n'

    main_str += ')'
    return main_str

  def __setattr__(self, key, value: 'Operator'):
    obj = getattr(self, key, None)
    if key != "forward" and obj is not None and callable(obj):
      raise AttributeError(f"cannot assign {key} as it is already a method")
    if isinstance(value, Operator):
      if not "_operators" in self.__dict__:
        raise AttributeError("cannot assign operator before Operator.__init__() call")
      if key in self.__dict__ and key not in self._operators:
        raise KeyError(f"attribute '{key}' already exists")
      self._operators[key] = value
    self.__dict__[key] = value

  # properties/

  def operators(self):
    r"""Returns an iterator over all operators in the job."""
    for _, module in self.named_operators():
      yield module

  def named_operators(self, memo = None, prefix: str = '', remove_duplicate: bool = True):
    r"""Returns an iterator over all modules in the network, yielding
    both the name of the module as well as the module itself."""
    if memo is None:
      memo = set()
    if self not in memo:
      if remove_duplicate:
        memo.add(self)
      yield prefix, self
      for name, module in self._operators.items():
        if module is None:
          continue
        submodule_prefix = prefix + ('.' if prefix else '') + name
        for m in module.named_operators(memo, submodule_prefix, remove_duplicate):
          yield m

  @property
  def children(self):
    return self._operators.values()

  # user can manually define inputs if they want, avoid this user if you don't know
  # how this system works
  _inputs = []
  @property
  def inputs(self):
    import inspect
    args = inspect.getfullargspec(self.forward).args
    try:
      args.remove('self')
    except:
      pass
    if self._inputs:
      args += self._inputs
    return args

  @property
  def state_dict(self) -> StateDictModel:
    return StateDictModel(
      state = self.__class__.__name__,
      data = {},
      inputs = self.inputs,
      outputs = self.outputs
    )

  # /properties

  # information passing/

  def propagate(self, **kwargs):
    for c in self.children:
      c.propagate(**kwargs)
    for k, v in kwargs.items():
      setattr(self, k, v)

  def thaw(self, flowchart):
    nodes = flowchart["nodes"]
    edges = flowchart["edges"]
    for n in nodes:
      name = n["name"]
      if name.startswith("self."):
        name = name[5:]
      if hasattr(self, name):
        op: 'Operator' = getattr(self, name)
        op.propagate(
          node_info = n,
          source_edges = list(filter(
            lambda x: x["target"] == n["id"], edges
          ))
        )

  # /information passing

  def forward(self):
    raise NotImplementedError("User must implement forward()")

  def _register_forward(self, python_callable: Callable):
    # convienience method to register a forward method
    self.forward = python_callable

  node_info = None
  source_edges = None

  def __call__(self, *args, **kwargs):
    # blank comment so docstring below is not loaded

    """There is no need to perform ``self.is_dag`` check here, since it is
    a declarative model not an imperative one, so existance of DAG's is
    irrelevant.

    This function will has the code for the following:
    1. Input checks, we want to enforce that
    2. Tracing, we want to trace the execution of the model
    3. Networking, when required to execute like RPC
    """

    # Type Checking and create input dicts
    inputs = self.inputs
    len_inputs = len(args) + len(kwargs)
    if len_inputs > len(inputs):
      raise ValueError(f"Number of arguments ({len(inputs)}) does not match number of inputs ({len_inputs})")
    elif len_inputs < len(args):
      raise ValueError(f"Need at least arguments ({len(args)}) but got ({len_inputs})")

    input_dict = {}
    for i, arg in enumerate(args):
      input_dict[self.inputs[i]] = arg
    for key, value in kwargs.items():
      if key in inputs:
        input_dict[key] = value

    if self.node_info != None:
      self.node_info["run_status"]["start"] = datetime.now().isoformat()
      self.node_info["run_status"]["inputs"] = {k: str(type(v)) for k, v in input_dict.items()}
      self._trace_object(self.node_info)

    # ----input
    out = self.forward(**input_dict) # pass this through the user defined forward()
    # ----output

    if self.node_info != None:
      outputs = {}
      if out == None:
        outputs = {"out_0": type(None)}
      elif isinstance(out, dict):
        outputs = {k: type(v) for k, v in out.items()}
      elif isinstance(out, (list, tuple)):
        outputs = {f"out_{i}": type(v) for i, v in enumerate(out)}
      else:
        outputs = {"out_0": type(out)}
      self.node_info["run_status"]["end"] = datetime.now().isoformat()
      self.node_info["run_status"]["outputs"] = outputs
      self._trace_object(self.node_info)

    return out

  # nbx/

  def deploy(
    self,
    workspace: str,
    init_folder: str,
    job: Union[str, int],
    schedule: Cron = None,
    cache_dir: str = None,
    *,
    _return_data = False
  ) -> Job:
    """_summary_

    Args:
        workspace (str): Name of the workspace to be a part of this
        init_folder (str, optional): Name the folder to zip
        job (Union[str, int], optional): Name or ID of the job
        schedule (Cron, optional): If ``None`` will run only once, so be careful
        cache_dir (str, optional): Folder where to put the zipped file, if None will be tempdir
        _return_data (bool, optional): Internal

    Returns:
        Job: Job object
    """
    logger.info(f"Deploying {self.__class__.__name__} -> '{job}'")
    
    # TODO: @yashbonde add job -> job id/name resolver support after revamp
    # TODO: @yashbonde add workspace -> workspace id/name resolver support after revamp
    # data = nbox_ws_v1.workspace.u(workspace).jobs() # get all the jobs for this user
    # data = list(filter(lambda x: x["id"] == job or x["name"] == job, data)) # filter by name
    # if not len(data):
    #   raise ValueError(f"No job found with name or id '{job}'")
    # if len(data) > 1:
    #   raise ValueError(f"Multiple jobs found with name or id '{job}', please enter job id")
    # this_job = data[0]
    # job_id = this_job["id"]
    # job_name = this_job["name"]

    job_id = job
    job_name = U.get_random_name(True).split("-")[0] if job_id == None else job

    # check if this is a valid folder or not
    if not os.path.exists(init_folder) or not os.path.isdir(init_folder):
      raise ValueError(f"Incorrect project at path: '{init_folder}'! nbox jobs new <name>")
    if os.path.isdir(init_folder):
      os.chdir(init_folder)
      if not os.path.exists("./exe.py"):
        raise ValueError(f"Incorrect project at path: '{init_folder}'! nbox jobs new <name>")
    else:
      raise ValueError(f"Incorrect project at path: '{init_folder}'! nbox jobs new <name>")

    # flowchart-alpha
    dag = get_nbx_flow(self.forward)
    try:
      from json import dumps
      dumps(dag) # check if works
    except:
      logger.error("Cannot perform pre-building, only live updates will be available!")
      logger.debug("Please raise an issue on chat to get this fixed")
      dag = {"flowchart": None, "symbols": None}

    for n in dag["flowchart"]["nodes"]:
      name = n["name"]
      if name.startswith("self."):
        name = name[5:]
      operator_name = "CodeBlock" # default
      cls_item = getattr(self, name, None)
      if cls_item and cls_item.__class__.__base__ == Operator:
        operator_name = cls_item.__class__.__name__
      n["operator_name"] = operator_name

    if schedule != None:
      logger.debug(f"Schedule: {schedule.get_dict}")

    data = {
      "dag": dag,
      "schedule": schedule.get_dict() if schedule != None else None,
      "job_id": job_id,
      "job_name": job_name,
      "created": datetime.now().isoformat(),
    }

    if _return_data:
      return data

    with open(U.join(init_folder, "meta.json"), "w") as f:
      f.write(dumps(data))

    # zip all the files folder
    all_f = U.get_files_in_folder(init_folder)
    all_f = [f[len(init_folder)+1:] for f in all_f] # remove the init_folder from zip

    zip_path = U.join(cache_dir if cache_dir else gettempdir(), "project.zip")
    logger.info(f"Zipping project to '{zip_path}'")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
      for f in all_f:
        zip_file.write(f)
    
    return deploy_job(zip_path = zip_path, schedule = schedule, data = data, workspace = workspace)

  # /nbx
