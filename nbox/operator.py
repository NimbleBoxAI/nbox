"""
``Operators`` is how you write NBX-Jobs. If you are familiar with pytorch, then usage is
exactly same, for others here's a quick recap:

.. code-block:: python

  class MyOperator(Operator):
    def __init__(self, a: int, b: str):
      super().__init__()
      self.a: int = a
      self.b: Operator = MyOtherOperator(b) # nested calling
    
    def forward(self, x: int) -> int:
      y = self.a + x
      y = self.b(y) + x # nested calling
      return y

  job = MyOperator(1, "hello") # define once
  res = job(2)                 # use like python, screw DAGs

We always wanted to ensure that there is least developer resistance in the way of using ``Operator``
so there is a convinient ``operator`` decorator that can wrap any function or class and extend all
the powerful methods available in the ``Operator`` object, like ``.deploy()``. By default every
wrapped function is run as a Job.

.. code-block:: python

  @operator()
  def foo(i: float = 4):
    return i * i

  # to deploy the operator 
  if __name__ == "__main__":
    # pass deployment_type = "serving" to make an API
    foo_remote = foo.deploy('workspace-id')
    assert foo_remote() == foo()
    assert foo_remote(10) == foo(10)

And you can make simple stateful object like classes using ``@operator`` decorator by making it
an API endpoint.

.. code-block:: python

  @operator()
  class Bar:
    def __init__(self, x: int = 1):
      self.x = x

    def inc(self):
      self.x += 1

    def getvalue(self):
      return self.x

    def __getattr__(self, k: str):
      # simple echo to demonstrate that underlying python object methods can
      # also be accessed over the internet
      return str

  if __name__ == "__main__":
    bar_remote = Bar.deploy('workspace-id')
    
    # increment the numbers
    bar.inc(); bar_remote.inc()

    # directly access the values, no need for futures
    assert bar.x == bar_remote.x

    print(bar.jj_guverner, bar_remote.jj_guverner)

If you want to use the APIs for deployed jobs and servings `nbox.Jobs <nbox.jobs.html>`_ is a better documentation.


Engineering
-----------

Fundamentally operators act as a wrapper on user code, sometime abstracting away functions
by breaking them into ``__init__``s and ``forward``s. But this is a simpler way to wrap
user function than letting users wrap their own function. It is easy to get false positives,
and so we explicitly expand things in two. These operators are like ``torch.nn.Modules``
spiritually as well because modules manage the underlying weights and operators manage the
underlying user logic.

Operators are combination of several subsystems that are all added in the same class, though
certainly if we come up with that high abstraction we will refactor this:

#. tree: All operators are really treated like a tree meaning that the execution is nested\
    and the order of execution is determined by the order of the operators in the tree. DAGs\
    are fundamentally just trees with some nodes spun togeather, to execute only once.
#. deploy, ...: All the services in NBX-Jobs.
#. get_nbx_flow: which is the static code analysis system to understand true user intent and\
    if possible (and permission of the user) optimise the logic.


Tips
----

``Operators`` are built to be the abstract equivalent of any computation so code can be easily
run in distributed fashion.

#. Use Operator directly as a function as much as possible, it's the simplest way to use it.
#. ``@operator`` decorator on your function and it will be run as a job by default, you want that.
#. ``@operator`` decorator on your class and it will be run as a serving by default, you want that.

Documentation
-------------
"""
# Some parts of the code are based on the pytorch nn.Module class
# pytorch license: https://github.com/pytorch/pytorch/blob/master/LICENSE
# modifications: research@nimblebox.ai 2022

import os
import re
import inspect
import requests
from enum import Enum
from time import sleep
from functools import partial
from collections import OrderedDict
from typing import Any, Dict, List, Tuple, Union

import nbox.utils as U
from nbox.auth import ConfigString, secret
from nbox.utils import logger
from nbox.nbxlib.tracer import Tracer
from nbox.version import __version__
from nbox.sub_utils.latency import log_latency
from nbox.framework.on_functions import get_nbx_flow
from nbox.framework import AirflowMixin, PrefectMixin
from nbox.hyperloop.job_pb2 import Job as JobProto, Resource
from nbox.hyperloop.dag_pb2 import DAG, Flowchart, Node, RunStatus
from nbox.network import _get_job_data
from nbox.jobs import Schedule, new as new_folder, Job, Serve
from nbox.messages import get_current_timestamp, write_binary_to_file
from nbox.relics import RelicsNBX
from nbox.init import nbox_ws_v1
from nbox.subway import SpecSubway


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


DEFAULT_RESOURCE = Resource(
  cpu = "100m",         # 100mCPU
  memory = "200Mi",     # MiB
  disk_size = "1Gi",    # GiB
  gpu = "none",         # keep "none" for no GPU
  gpu_count = "0",      # keep "0" when no GPU
  timeout = 120_000,    # 2 minutes between attempts
  max_retries = 3,      # third times the charm :P
)

# we will keep on expanding this list, note that this cannot be directly used with copytree,
# to make it work remove the trailing slash
FN_IGNORE = [
  "__pycache__/", "venv/", ".git/", ".vscode/"
]


class Operator():
  node = Node()
  source_edges: List[Node] = None
  _op_type = OperatorType.UNSET
  _op_wrap = None
  _op_wrap_init = None

  # this is a map from operator to resource, by deafult the values will be None,
  # unless explicitly modified by `chief.machine.Machine`
  _current_resource = None
  _op_to_resource_map = {}

  def __init__(self) -> None:
    """Create an operator, which abstracts your code into sharable, bulding blocks which
    can then deployed on either NBX-Jobs or NBX-Deploy.
    
    Usage:

    .. code-block:: python
      
      class MyOperator(Operator):
        def __init__(self, ...):
          ... # initialisation of job happens here

          # use prebuilt operators to define the entire process
          from nbox.lib.shell import Shell
          self.download_binary = Shell("wget https://nbox.ai/{hash}")

          # keep a library of organisation wide operators
          from .nbx_internal.operators import TrainGPT
          self.train_model: Operator = TrainGPT(...)

        def forward(self, ...):
          # pass any inputs you want at runtime
          ... # execute code in any arbitrary order, free from DAGs

          self.download_binary(hash="my_binary") # pass relevant data at runtime
          self.train_model() # run any operation and get full visibility on platform

      # to convert operator is 
      job: Operator = MyOperator(...)

      # deploy this as a batch process or API endpoint or API endpoint
      job.deploy() # WIP # WIP
    """
    self._operators: Dict[str, 'Operator'] = OrderedDict() # {name: operator}
    self._op_trace = []
    self._tracer: Tracer = None

  def __remote_init__(self):
    """User can overwrite this function, this will be called only when running on remote.
    This helps in with things like creating the models can caching them in self, instead
    of ``lru_cache`` in forward."""
    pass

  def remote_init(self):
    """Triggers `__remote_init__` across the entire tree."""
    self.__remote_init__()
    for _, op in self._operators.items():
      op.remote_init()

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

    main_str = self.__qualname__ + '('
    if lines:
      # simple one-liner info, which most builtin Modules will use
      if len(extra_lines) == 1 and not child_lines:
        main_str += extra_lines[0]
      else:
        main_str += '\n  ' + '\n  '.join(lines) + '\n'

    main_str += ')'
    return main_str

  # mixin/

  to_airflow_operator = AirflowMixin.to_airflow_operator
  to_airflow_dag = AirflowMixin.to_airflow_dag
  from_airflow_operator = classmethod(AirflowMixin.from_airflow_operator)
  from_airflow_dag = classmethod(AirflowMixin.from_airflow_dag)
  to_prefect_task = PrefectMixin.to_prefect_task
  to_prefect_flow = PrefectMixin.to_prefect_flow
  from_prefect_task = classmethod(PrefectMixin.from_prefect_task)
  from_prefect_flow = classmethod(PrefectMixin.from_prefect_flow)

  # Operators are very versatile and can be created in many different ways, heres a list of those:
  # 1. from an existing job, where calling it triggers a new run and args are passed through relics
  # 2. from an existing serving, where calling it equates to an HTTP API call
  # 3. from a function, where we decorate an existing function and convert it to an operator

  @classmethod
  def from_job(cls, job_id_or_name, workspace_id: str):
    """latch an existing job so that it can be called as an operator."""
    # implement this when we have the client-server that allows us to get the metadata for the job
    if not workspace_id:
      raise DeprecationWarning("Personal workspace does not support serving")
    job_id, job_name = _get_job_data(job_id_or_name, workspace_id = workspace_id)
    if job_id is None:
      raise ValueError(f"No serving found with name {job_name}")

    logger.debug(f"Latching to job '{job_name}' ({job_id})")
    
    def forward(*args, **kwargs):
      """This is the forward method for a NBX-Job. All the parameters will be passed through Relics."""
      logger.debug(f"Running job '{job_name}' ({job_id})")
      job = Job(job_id, workspace_id = workspace_id)
      relic = RelicsNBX("dot_deploy_cache", workspace_id, create = True)
      
      # determining the put location is very tricky because there is no way to create a sync between the
      # key put here and what the run will pull. This can lead to many weird race conditions. So for now
      # I am going to rely on the fact that we cannot have two parallel active runs. Thus at any given
      # moment there can be only one file at /{job_id}/args_kwargs.pkl
      relic.put_object(f"{job_id}/args_kwargs", (args, kwargs))

      # and then we will trigger the job and wait for the run to complete
      job.trigger()
      latest_run = job.last_n_runs(1)
      logger.debug(f"New run ID: {latest_run['id']}")
      max_retries = latest_run['resource']['max_retries']

      max_polls = 600  # about 10 mins
      poll_count = 0
      while latest_run["retry_count"] <= max_retries:
        # create a polling loop
        logger.debug(f"[{poll_count:04d}/{max_polls:04d}] [{latest_run['retry_count']}/{max_retries}] {latest_run['status']}")
        if poll_count > max_polls:
          raise TimeoutError(f"Run {latest_run['id']} timed out after {max_polls} polls")
        if latest_run["status"] == "COMPLETED":
          break
        elif latest_run["retry_count"] == max_retries and latest_run["status"] == "ERROR":
          raise RuntimeError(f"Run {latest_run['id']} failed after {max_retries} retries")
        latest_run = job.last_n_runs(1)
        poll_count += 1
        sleep(1)

      if latest_run["status"] == "ERROR":
        raise Exception(f"Run failed after {max_retries} retries")

      # assuming everything went well we should have a file at /{job_id}/return
      obj = relic.get_object(f"{job_id}/return")
      return obj
    
    # create the class and override some values to make more sense
    _op = cls()
    _op.__qualname__ = "job_" + job_id
    _op.forward = forward
    _op._op_type = OperatorType.JOB

    return _op

  @classmethod
  def from_serving(cls, url: str, token: str):
    """Latch to an existing serving operator

    Args:
      url (str): The URL of the serving
      token (str): The token to access the deployment, get it from settings.
    """
    logger.debug(f"Latching to serving: {url}")

    # now we can run a NBX-Let either on a Pod or on the Build instance, so we can check the URL
    # once and make a judgement based on that

    session = requests.Session()
    if re.match("https:\/\/api\.nimblebox\.ai\/(\w+)\/", url):
      # this is deployment on a Pod
      session.headers.update({"NBX-KEY": token})
    elif re.match("https:\/\/(\w+)-(\w+)\.build([\.rc]+)*\.nimblebox\.ai\/", url):
      # this is deployment on a Build instance, there's a catch though without knowing the
      session.headers.update({
        "NBX-TOKEN": token,
        "X-NBX-USERNAME": secret.get("username"),
      })
    else:
      raise ValueError(f"Invalid URL: {url}")

    openapi_url = f"{url}openapi.json" # url has as trailing slash
    r = session.get(openapi_url)
    try:
      data = r.json()
    except:
      print(r.content)
      raise

    REQ = type("REQ", (object,), {})

    # create a function to input spec mapper and so the generic forward method can be 
    fn_spec = {}
    for p, v in data["paths"].items():
      if p.startswith("/method_"):
        fn_name = p[8:]
        v = v["post"]
        # op_id = v["operationId"]
        ref_path = v["requestBody"]["content"]["application/json"]["schema"]["$ref"].split("/")[1:]
        out = data
        for p in ref_path:
          out = out[p]
        args = out["properties"]
        fn_spec[fn_name] = args
      elif p == "/forward":
        fn_name = "forward"
        v = v["post"]
        # op_id = v["operationId"]
        ref_path = v["requestBody"]["content"]["application/json"]["schema"]["$ref"].split("/")[1:]
        out = data
        for p in ref_path:
          out = out[p]
        args = out["properties"]
        fn_spec[fn_name] = args

    serving_stub = SpecSubway.from_openapi(data, _url = url, _session = session)

    # define the forward function for this serving operator, the objective is that this will be able to handle
    # args, kwargs just like how it works on the local machine
    def forward(method, *args, **kwargs):
      if method in fn_spec:
        # check in the kwargs if we have any arguments to pass
        args_dict = {k: v.get("default", REQ) for k, v in fn_spec[method].items()}
        _data = {} # the json that will be sent over
        for i, (k, v) in enumerate(args_dict.items()):
          if len(args) == i:
            break
          _data[k] = args[i]
        for k, v in kwargs.items():
          _data[k] = v

        # client-side check for unknown vars
        unknown_args = set()
        for k, v in _data.items():
          if k not in args_dict:
            unknown_args.add(k)
        if len(unknown_args):
          raise ValueError(f"Unknown arguments: {unknown_args}")

        missing_args = set()
        for k, v in args_dict.items():
          if k not in _data and v == REQ:
            missing_args.add(k)
        if len(missing_args):
          raise ValueError(f"Missing required arguments: {missing_args}")

        # this is a simple method call
        if method == "forward":
          fn = "forward"
        elif method in fn_spec:
          fn = f"method_{method}"

      else:
        _data = {"rpc_name": method, "key": U.py_to_bs64(args[0])}
        if len(args) > 1:
          _data["value"] = U.py_to_bs64(args[1])
        fn = "nbx_py_rpc"

      # now we can call the function
      data = serving_stub.u(fn)(**_data)
      
      # convert to usable values
      if not data["success"]:
        raise Exception(data["message"])
      value = U.py_from_bs64(data["value"])
      return value

    # call the stub and get details of the operator
    try:
      data = serving_stub.who_are_you()
    except AttributeError as e:
      logger.error(f"Error: {e}")
      logger.error("Unable to connect to the serving, you are probably not connected to a nbox serving")
      raise ValueError("Unable to connect to the serving, you are probably not connected to a nbox serving")


    # create the class and override some values to make more sense
    _op = cls()
    _op.propagate(_nbx_serving_fn_spec = fn_spec)
    _op.__qualname__ = "serving_" + data["name"]
    _op.forward = forward
    _op._op_type = OperatorType.SERVING
    return _op

  @classmethod
  def fn(cls):
    """Wraps the function as an Operator, so you can use all the same methods as Operator"""
    def wrap(fn):
      if type(fn) == type(wrap): # lol the quick hack
        # this is a wrapped function to be run as a job
        op = cls()
        op = op._fn(fn)
        return op
      elif type(fn) == type(Operator):
        # this is a little bit tricky since the initialisation of the object has be be done
        # by the user later and thus we wrap another function which actually initialises the
        # object
        def cls_init(*args, **kwargs):
          op = cls()
          op = op._cls(fn, *args, **kwargs)
          op._op_wrap_init = (args, kwargs)
          return op
        return cls_init
    return wrap

  def _cls(self, fn, *args, **kwargs):
    """Do not use directly, use ``@operator`` decorator instead. Utility to wrap a class as an operator"""
    obj = fn(*args, **kwargs)
    # override the file this is necessary for getting the remote operations running
    self.__file__ = inspect.getfile(fn)
    self.__doc__ = obj.__doc__
    self.__qualname__ = "cls_" + obj.__class__.__qualname__
    self._op_type = OperatorType.WRAP_CLS
    self._op_wrap = obj
    return self

  def _fn(self, fn):
    """Do not use directly, use ``@operator`` decorator instead. Utility to wrap a function as an operator"""
    self.forward = fn # override the forward function
    # override the file this is necessary for getting the remote operations running
    self.__file__ = inspect.getfile(fn)
    self.__doc__ = fn.__doc__
    self.__qualname__ = "fn_" + fn.__name__
    self._op_type = OperatorType.WRAP_FN
    self._op_wrap = fn
    return self

  # /mixin

  # python state modification /

  """The functions below are part of a larger effort to make Operator interact better with the underlying
  user classes and functions. Most Importantly we are borrowing the ray concept that functions map to jobs
  and classes map to actors. We are also extending it a little bit to make it more pythonic in ways by adding
  suppoprt for multiple inbuilt python object methods like __getitem__, __setitem__ etc. So in order to make
  it work we are adding those methods below as well and each method will based on its type take a call on
  what to do.
  """

  def __setattr__(self, key, value: 'Operator'):
    obj = getattr(self, key, None)
    if key != "forward" and obj is not None and callable(obj) and not isinstance(value, Operator):
      raise AttributeError(f"cannot assign {key} as it is already a method")
    if isinstance(value, Operator):
      if not "_operators" in self.__dict__:
        raise AttributeError("cannot assign operator before super().__init__() call")
      if key in self.__dict__ and key not in self._operators:
        raise KeyError(f"attribute '{key}' already exists")
      self._operators[key] = value

      # map the operator to the current resource
      self._op_to_resource_map[key] = self._current_resource
    self.__dict__[key] = value

  def __getattr__(self, key):
    if self._op_type == OperatorType.SERVING:
      if key in self._nbx_serving_fn_spec:
        return partial(self.forward, key)
      return self.forward("__getattr__", key)
    elif self._op_type == OperatorType.WRAP_CLS:
      return getattr(self._op_wrap, key)
    elif self._op_type == OperatorType.UNSET and key == "__qualname__":
      return self.__class__.__qualname__
    raise AttributeError(f"{key}")

  def __getitem__(self, key):
    if self._op_type == OperatorType.SERVING:
      return self.forward("__getitem__", key)
    if self._op_type in [OperatorType.WRAP_FN, OperatorType.WRAP_CLS]:
      return self._op_wrap[key]
    raise KeyError(f"{key}")

  def __setitem__(self, key, value):
    if self._op_type == OperatorType.SERVING:
      return self.forward("__setitem__", key, value)
    if self._op_type in [OperatorType.WRAP_CLS]:
      self._op_wrap[key] = value
      return
    raise KeyError(f"{key}")

  def __delitem__(self, key):
    if self._op_type == OperatorType.SERVING:
      return self.forward("__delitem__", key)
    if self._op_type in [OperatorType.WRAP_CLS]:
      del self._op_wrap[key]; return
    raise KeyError(f"{key}")

  def __iter__(self):
    if self._op_type == OperatorType.SERVING:
      return self.forward("__iter__")
    if self._op_type in [OperatorType.WRAP_CLS]:
      return iter(self._op_wrap)
    raise ValueError(f"Operator cannot iterate")

  def __next__(self):
    if self._op_type == OperatorType.SERVING:
      return self.forward("__next__")
    if self._op_type in [OperatorType.WRAP_CLS]:
      return next(self._op_wrap)
    raise ValueError(f"Operator cannot iterate")

  def __len__(self):
    if self._op_type == OperatorType.SERVING:
      return self.forward("__len__")
    if self._op_type in [OperatorType.WRAP_CLS]:
      return len(self._op_wrap)
    raise ValueError(f"Operator cannot iterate")

  def __contains__(self, key):
    if self._op_type == OperatorType.SERVING:
      return self.forward("__contains__", key)
    if self._op_type in [OperatorType.WRAP_CLS]:
      return key in self._op_wrap
    raise ValueError(f"Operator cannot iterate")

  # / python state modification

  def propagate(self, **kwargs):
    """Set kwargs for each child in the Operator"""
    for k, v in kwargs.items():
      setattr(self, k, v)
    for c in self._operators.values():
      c.propagate(**kwargs)

  def thaw(self, job: JobProto):
    """Load JobProto into this Operator"""
    nodes = job.dag.flowchart.nodes
    edges = job.dag.flowchart.edges
    for _id, node in nodes.items():
      name = node.name
      if name.startswith("self."):
        name = name[5:]
      if hasattr(self, name):
        op: 'Operator' = getattr(self, name)
        op.propagate(
          node = node,
          source_edges = list(filter(
            lambda x: edges[x].target == node.id, edges.keys()
          ))
        )

  def _named_operators(self, memo = None, prefix: str = '', remove_duplicate: bool = True):
    r"""Returns an iterator over all modules in the network, yielding both the name of the module
    as well as the module itself."""
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
        for m in module._named_operators(memo, submodule_prefix, remove_duplicate):
          yield m

  def __call__(self, *args, **kwargs):
    # check if calling is allowed based on the type
    if self._op_type == OperatorType.SERVING:
      # the fn_spec is set during the .from_serving() classmethod
      if len(self._nbx_serving_fn_spec) == 1 and "forward" in self._nbx_serving_fn_spec:
        # this means that an OperatorType.UNSET or OperatorType.WRAP_FN are on the other side of the pipe
        return self.forward("forward", *args, **kwargs)
      else:
        # this means that OperatorType.WRAP_CLS is on the other side of the pipe
        raise ValueError(f"Cannot call servings directly, will interfere with nbox")
    elif self._op_type == OperatorType.WRAP_CLS:
      raise ValueError(f"Cannot call class wrappers directly, will interfere with nbox")

    input_dict = {}
    logger.debug(f"Calling operator '{self.__class__.__name__}': {self.node.id}")
    _ts = get_current_timestamp()
    self.node.run_status.CopyFrom(RunStatus(start = _ts, inputs = {k: str(type(v)) for k, v in input_dict.items()}))
    if self._tracer != None:
      self._tracer(self.node)
    # ---- USER SEPERATION BOUNDARY ---- #

    with log_latency(f"{self.__class__.__name__}-forward"):
      out = self.forward(*args, **kwargs)

    # ---- USER SEPERATION BOUNDARY ---- #
    outputs = {}
    logger.debug(f"Ending operator '{self.__class__.__name__}': {self.node.id}")
    _ts = get_current_timestamp()
    self.node.run_status.MergeFrom(RunStatus(end = _ts, outputs = {k: str(type(v)) for k, v in outputs.items()}))
    if self._tracer != None:
      self._tracer(self.node)
    return out

  def forward(self):
    raise NotImplementedError("User must implement forward()")

  # nbx/
  def _get_dag(self) -> DAG:
    """Get the DAG for this Operator including all the nested ones."""
    dag = get_nbx_flow(self.forward)
    all_child_nodes = {}
    all_edges = {}
    for child_id, child_node in dag.flowchart.nodes.items():
      name = child_node.name
      if name.startswith("self."):
        name = name[5:]
      operator_name = "CodeBlock" # default
      cls_item = getattr(self, name, None)
      if cls_item is not None and cls_item.__class__.__base__ == Operator:
        # this node is an operator
        operator_name = cls_item.__class__.__name__
        child_dag: DAG = cls_item._get_dag() # call this function recursively

        # update the child nodes with parent node id
        for _child_id, _child_node in child_dag.flowchart.nodes.items():
          if _child_node.parent_node_id == "":
            _child_node.parent_node_id = child_id # if there is already a parent_node_id set is child's child
          all_child_nodes[_child_id] = _child_node
          all_edges.update(child_dag.flowchart.edges)
      
      # update the child nodes name with the operator name
      child_node.operator = operator_name

    # update the root dag with new children and edges
    _nodes = {k:v for k,v in dag.flowchart.nodes.items()}
    _edges = {k:v for k,v in dag.flowchart.edges.items()}
    _nodes.update(all_child_nodes)
    _edges.update(all_edges)

    # because updating map in protobuf is hard
    dag.flowchart.CopyFrom(Flowchart(nodes = _nodes, edges = _edges))
    return dag

  def deploy(
    self,
    workspace_id: str,
    id_or_name:str = None,
    deployment_type: str = None,
    resource: Resource = None,
    ignore_patterns: List[str] = [],
    *,
    _unittest = False,
    _include_pattern = [],
  ) -> 'Operator':
    """Uploads relevant files to the cloud and deploys as a batch process or and API endpoint, returns the relevant
    ``.from_job()`` or ``.from_serving`` Operator. This uploads the entire folder where the caller file is located.
    In which case having a ``.nboxignore`` and ``requirements.txt`` will also be moved over.

    Args:
      workspace_id (str): The workspace id to deploy to.
      id_or_name (str, optional): The id or name of the deployment. if deployment_type is 'serving' this this must be provided.
      deployment_type (str, optional): Defaults to 'serving' if WRAP_CLS else 'job'. The type of deployment to create.
      resource (Resource, optional): The resource to deploy to, uses a reasonable default.
    """
    # go over reasonable checks for deployment
    if deployment_type == None:
      if self._op_type in [OperatorType.WRAP_CLS, OperatorType.SERVING]:
        deployment_type = "serving"
      else:
        deployment_type = "job"
    if self._op_type == OperatorType.WRAP_CLS and deployment_type != "serving":
      raise ValueError(f"Cannot deploy a class as a job, only as a serving")
    if deployment_type not in OperatorType._valid_deployment_types():
      raise ValueError(f"Invalid deployment type: {deployment_type}. Must be one of {OperatorType._valid_deployment_types()}")
    if deployment_type == OperatorType.SERVING.value:
      if id_or_name is None:
        raise ValueError("id_or_name must be provided for serving deployment")

    # get the filepath and name to import for convience
    if self._op_type == OperatorType.UNSET:
      fp = inspect.getfile(self.__class__)
      name = self.__class__.__qualname__
    elif self._op_type in [OperatorType.JOB, OperatorType.SERVING]:
      raise ValueError("Cannot deploy an operator that is already deployed")
    elif self._op_type == OperatorType.WRAP_FN:
      fp = self.__file__
      name = self.__qualname__[3:] # to account for "fn_"
    elif self._op_type == OperatorType.WRAP_CLS:
      fp = self.__file__
      name = self.__qualname__[4:] # to account for "cls_"
    fp = os.path.abspath(fp) # get the abspath, will be super useful later
    folder, file = os.path.split(fp)
    logger.info(f"Deployment Type: {deployment_type}")
    logger.info(f"Deploying '{name}' from '{fp}'")
    logger.info(f"Will upload folder: {folder}")

    # create a temporary directory to store all the files
    # copy over all the files and wait for it, else the changes below won't be reflected
    # dude damn, how did I not know this: https://dev.to/ackshaey/macos-vs-linux-the-cp-command-will-trip-you-up-2p00

    # so basically till now all we know is that `from fp import name` and `name()`. This process can now
    # be used to create a custom `nbx_user.py` file from which `exe.py` can import the three functions
    # and use them the conventional way. the three functions could simply be a router and the real operator
    # can be imported from the orginal file.
    init = ''
    if self._op_type in [OperatorType.UNSET, OperatorType.WRAP_CLS]:
      init = '()'

    with open(U.join(folder, "nbx_user.py"), "w") as f:
      f.write(f'''# Autogenerated for .deploy() call
from nbox.messages import read_file_to_binary
from nbox.hyperloop.job_pb2 import Resource

from {file.split('.')[0]} import {name}

def get_op(*_, **__):
  # takes no input since programtically generated returns the exact object
  out = {name}{init}
  return out

get_resource = lambda: read_file_to_binary('.nbx_core/resource.pb', message = Resource())
get_schedule = lambda: None
''')

    # create a requirements.txt file if it doesn't exist with the latest nbox version
    if not os.path.exists(U.join(folder, "requirements.txt")):
      with open(U.join(folder, "requirements.txt"), "w") as f:
        f.write(f'nbox[serving]=={__version__}\n')

    # create a .nboxignore file if it doesn't exist, this will tell all the files not to be uploaded to the job pod
    _igp = set(FN_IGNORE + ignore_patterns)
    _igp -= set(_include_pattern)
    if not os.path.exists(U.join(folder, ".nboxignore")):
      with open(U.join(folder, ".nboxignore"), "w") as f:
        f.write("\n".join(_igp))
    else:
      with open(U.join(folder, ".nboxignore"), "r") as f:
        _igp = _igp.union(set(f.read().splitlines()))
      with open(U.join(folder, ".nboxignore"), "w") as f:
        f.write("\n".join(_igp))

    # just create a resource.pb, if it's empty protobuf will work it out
    nbx_folder = U.join(folder, ".nbx_core")
    os.makedirs(nbx_folder, exist_ok = True)
    write_binary_to_file(resource or DEFAULT_RESOURCE, file = U.join(nbx_folder, "resource.pb"))

    if _unittest:
      return

    if deployment_type == OperatorType.JOB.value:
      out = Job.upload(
        init_folder = folder,
        id_or_name = id_or_name or "dot_deploy_runs", # all the .deploy() methods will be called dot_deploy_runs
        workspace_id = workspace_id,
      )
      return self.from_job(
        job_id_or_name = out.id,
        workspace_id = workspace_id,
      )
    elif deployment_type == OperatorType.SERVING.value:
      out = Serve.upload(init_folder = folder, id_or_name = id_or_name, workspace_id = workspace_id)

      # get the serving object
      stub = nbox_ws_v1.workspace.u(workspace_id).deployments.u(out.id)
      data = stub()
      token = data["deployment"]["api_key"]
      api_url = data["deployment"]["api_url"]
      model_url = api_url + out.model_id + "/"
      logger.info("model_url: " + model_url)

      # we will poll here till the model is not ready and return the latched operator
      done = False
      while not done:
        sleep(1)
        data = stub()
        this_model = list(filter(lambda x: x["id"] == out.model_id, data["models"]))[0]
        status = this_model["status"]
        logger.debug(f"status of model: {status}")
        if status == "deployment.failed":
          raise Exception("Deployment failed")
        done = status == "deployment.ready"

      return self.from_serving(model_url, token = token)


  def map(self, inputs: Union[List[Any], Tuple[Any]]):
    pass

  # /nbx

class Machine():
  def __init__(self, parent_op: Operator, resource: Resource):
    """Simple class to scope the operations to a specific resource"""
    self.parent_op = parent_op
    self.resource = resource

    # these are all the ops that are to be run on this machine as a group
    self.ops_list = []

  def add_child(self, op: Operator):
    self.ops_list.append(op)

  def __enter__(self):
    self.parent_op._current_resource = self.resource

  def __exit__(self, *args):
    self.parent_op._current_resource = None


operator = Operator.fn # convinience
