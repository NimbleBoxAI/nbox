# Some parts of the code are based on the pytorch nn.Module class
# pytorch license: https://github.com/pytorch/pytorch/blob/master/LICENSE
# due to requirements of stability, some type enforcing is performed

import os
from typing import Callable
from functools import partial
from tempfile import gettempdir
from collections import OrderedDict
from datetime import datetime, timedelta

from logging import getLogger
logger = getLogger()

from ..network import deploy_job
from ..utils import join
from ..framework.__airflow import AirflowMixin
from ..framework.on_functions import get_nbx_flow, DBase


class StateDictModel(DBase):
  __slots__ = [
    "state", # :str
    "data", # :dict
    "inputs", # :dict
    "outputs", # :dict
  ]


# class TraceObject:
#   def __init__(self, root):
#     self.root = root
#     self.flow = OrderedDict()
#
#   def pre(self, inputs, cls):
#     _id = id(cls)
#     self.flow[_id] = {
#       "id": _id,
#       "class_name": cls.__class__.__name__,
#       "inputs": {k: type(v) for k, v in inputs.items()},
#       "outputs": {},
#       "start": datetime.now(),
#       "end": None,
#     }
#
#   def post(self, out, cls):
#     _id = id(cls)
#     if _id not in self.flow:
#       raise ValueError(f"{_id} not found in flow")
#
#     outputs = {}
#     if out == None:
#       outputs = {"out_0": type(None)}
#     elif isinstance(out, dict):
#       outputs = {k: type(v) for k, v in out.items()}
#     elif isinstance(out, (list, tuple)):
#       outputs = {f"out_{i}": type(v) for i, v in enumerate(out)}
#     else:
#       outputs = {"out_0": type(out)}
#
#     self.flow[_id].update({
#       "outputs": outputs,
#       "end": datetime.now(),
#     })
#     self.flow[_id]["duration"] = self.flow[_id]["end"] - self.flow[_id]["start"]
#
#     # convert datetime to string objects for serialization
#     self.flow[_id]["start"] = self.flow[_id]["start"].isoformat()
#     self.flow[_id]["end"] = self.flow[_id]["end"].isoformat()
#     self.flow[_id]["duration"] = str(self.flow[_id]["duration"].total_seconds())
#
#   def to_dict(self):
#     return {
#       "root": self.root.__class__.__name__,
#       "root_id": id(self.root),
#       "flow": self.flow,
#     }
#
#   def dag(self, depth = 1, root_ = None):
#     if depth != 1:
#       raise ValueError("depth of 1 supported only")
#
#     if depth < 0:
#       return []
#
#     dag = []
#     # create nodes
#     root_ = root_ if root_ != None else self.root
#     for _name, c in root_._operators.items():
#       _id = id(c)
#       name = c.__class__.__name__
#       _trace = self.flow[_id]
#       _trace["code_name"] = _name
#
#       # if there is some depth left, recurse
#       if depth > 1:
#         children_dag = self.dag(depth - 1, c)
#         _trace["children"] = children_dag
#
#       dag.append({
#         "id": _id,
#         "type": "input" if len(dag) == 0 else None,
#         "data": {
#           "label": f"{_name} | {name}"
#         },
#         # "meta": _trace
#       })
#     dag[-1]["type"] = "output"
#
#     # create edges
#     for src, trg in zip(dag[:-1], dag[1:]):
#       dag.append({
#         "id": f"edge-{src['id']}-{trg['id']}",
#         "source": src["id"],
#         "target": trg["id"]
#       })
#   
#     return dag

class Tracer:
  def __init__(self):
    try:
      # when job is running on NBX, gRPC stubs are used
      import nbox_js_stub
      self.l = nbox_js_stub.Trace()
      self._trace_obj = "stub"
    except ImportError:
      def _trace(x, fn):
        logger.info(x, extra={"fn": fn})

      self.l = _trace
      self._trace_obj = "logger"

  def __getattr__(self, __name):
    item = getattr(self.l, __name, None)
    if self._trace_obj == "logger":
      return partial(self.l, fn=__name)
    elif self._trace_obj == "stub" and item:
      return item
    raise AttributeError(f"Service: '{__name}' not found")

  def __call__(self, dag_update):
    # import requests
    # r = requests.post(
    #   url = "127.0.0.1:8000/log",
    #   json = dag_update,
    # )
    # print(r.content)
    print(dag_update)


class Operator(AirflowMixin):
  _version: int = 1

  def __init__(self) -> None:
    self._operators = OrderedDict() # {name: operator}
    self._op_trace = []

  # classmethods/

  def serialise(self):
    pass

  @classmethod
  def deserialise(cls, state_dict):
    pass

  # AirflowMixin methods
  # --------------------
  # AirflowMixin.to_airflow_operator(self, timeout, **operator_kwargs):
  # AirflowMixin.from_airflow_operator(cls, air_operator) <- classmethod
  # AirflowMixin.to_airflow_dag(self, dag_kwargs, operator_kwargs)
  # AirflowMixin.from_airflow_dag(cls, dag) <- classmethod

  # /classmethods

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

  # properties as planned

  @property
  def comms(self):
    # this returns all the things communications such as email, Slack, Discord
    # phone alerts, system notifications, gmeets, basically everything supported
    # in https://github.com/huggingface/knockknock/tree/master/knockknock
    raise NotImplementedError()

  # /properties

  # information passing/

  def propagate(self, **kwargs):
    for c in self.children:
      c.propagate(**kwargs)
    for k, v in kwargs.items():
      setattr(self, k, v)

  # /information passing

  def forward(self):
    raise NotImplementedError("User must implement forward()")

  def _register_forward(self, python_callable: Callable):
    # convienience method to register a forward method
    self.forward = python_callable

  _trace_object = Tracer()
  node_info = None
  source_edges = None

  # def trace(self, *args, return_dag = True, **kwargs):
  #   self._trace_object = TraceObject(self)
  #   self.propagate(_trace_object = self._trace_object)
  #   self(*args, **kwargs)
  #   trace = self._trace_object.to_dict()
  #   dag = self._trace_object.dag()
  #   self.propagate(_trace_object = None)
  #   self._trace_object = None
  #
  #   output = (trace,)
  #   if return_dag:
  #     output += (dag,)
  #   return output

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

  def thaw(self, flowchart):
    nodes = flowchart["nodes"]
    edges = flowchart["edges"]
    for n in nodes:
      name = n["name"]
      if name.startswith("self."):
        name = name[5:]
      if hasattr(self, name):
        op = getattr(self, name)
        op.propagate(
          node_info = n,
          source_edges = list(filter(
            lambda x: x["target"] == n["id"], edges
          ))
        )

  def deploy(
    self,
    init_folder: str = None,
    cache_dir: str = None,
    job_id = None,
    job_name = None,
    start_datetime: datetime = None,
    end_datetime: datetime = None,
    time_interval: timedelta = None,
  ):
    logger.info(f"Deploying {self.__class__.__name__} -> '{job_id}/{job_name}'")

    # flowchart-alpha
    dag = get_nbx_flow(self.forward)
    try:
      from json import dumps
      dumps(dag)
    except:
      logger.error("Cannot perform pre-building, only live updates will be available!")
      logger.info("Please raise an issue on chat to get this fixed")
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

    schedule_meta = {
      "start_datetime": start_datetime,
      "end_datetime": end_datetime,
      "time_interval": time_interval,
      "job_id": job_id,
      "job_name": job_name,
      "dag": dag,
      "created": datetime.now().isoformat(),
    }
    # print(schedule_meta)
    return schedule_meta

    # check if this is a valif folder or not
    if not os.path.exists(init_folder) or not os.path.isdir(init_folder):
      raise ValueError(f"Incorrect project at path: '{init_folder}'! nbox jobs init <name>")
    if os.path.isdir(init_folder):
      os.chdir(init_folder)
      if not os.path.exists("./exe.py"):
        raise ValueError(f"Incorrect project at path: '{init_folder}'! nbox jobs init <name>")
    else:
      raise ValueError(f"Incorrect project at path: '{init_folder}'! nbox jobs init <name>")

    # zip the folder
    import zipfile
    zip_path = join(cache_dir if cache_dir else gettempdir(), "project.zip")
    logger.info(f"Zipping project to '{zip_path}'")
    zip_file = zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED)
    for root, dirs, files in os.walk(init_folder):
      for file in files:
        zip_file.write(os.path.join(root, file))
    zip_file.close()

    return schedule_meta

    # deploy_job(zip_path = zip_path, schedule_meta = schedule_meta)

  # /nbx
