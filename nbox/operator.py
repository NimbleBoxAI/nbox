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

If you want to use deploy `nbox.Jobs <nbox.jobs.html>`_ is a better documentation.


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
"""
# Some parts of the code are based on the pytorch nn.Module class
# pytorch license: https://github.com/pytorch/pytorch/blob/master/LICENSE
# modifications: research@nimblebox.ai 2022

import os
import inspect
from time import sleep
import requests
from enum import Enum
from typing import Dict, Iterable, List, Union
from collections import OrderedDict
from subprocess import Popen

from nbox.utils import logger, env
from nbox.nbxlib.tracer import Tracer
from nbox.version import __version__
from nbox.sub_utils.latency import log_latency
from nbox.framework.on_functions import get_nbx_flow
from nbox.framework import AirflowMixin, PrefectMixin
from nbox.hyperloop.job_pb2 import Job as JobProto, Resource
from nbox.hyperloop.dag_pb2 import DAG, Flowchart, Node, RunStatus
from nbox.network import deploy_job, deploy_serving, _get_deployment_data, _get_job_data
from nbox.jobs import Schedule, new as new_folder, Job, Serve
from nbox.messages import get_current_timestamp, write_binary_to_file
from nbox.relics import RelicsNBX


class OperatorType(Enum):
  """This Enum does not concern the user, however I am describing it so people can get a feel of the breadth
  of what nbox can do. The purpose of ``Operator`` is to build an abstract representation of any possible compute
  and execute them in any fashion needed to improve the overall performance of any distributed software system.
  Here are the different types of operators:

  #. ``UNSET``: this is the default mode and is like using vanilla python without any nbox features.
  #. ``JOB``: In this case the process is run as a batch process and the I/O of values is done using Relics
  #. ``SERVING``: In this case the process is run as an API proces
  #. ``WRAP``: When we wrap a function as an ``Operator``
  """
  UNSET = "unset" # default
  JOB = "from_job"
  SERVING = "from_serving"
  WRAP = "function_wrap"


DEFAULT_RESOURCE = Resource(
  cpu = "100m",         # 100mCPU
  memory = "200Mi",     # MiB
  disk_size = "1Gi",    # GiB
  gpu = "none",         # keep "none" for no GPU
  gpu_count = "0",      # keep "0" when no GPU
  timeout = 120_000,    # 2 minutes between attempts
  max_retries = 3,      # third times the charm :P
)


class Operator():
  _version: int = 1 # always try to keep this an i32
  node = Node()
  source_edges: List[Node] = None
  _inputs = []
  _raise_on_io = True
  _op_type = OperatorType.UNSET

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
    job_id, job_name = _get_job_data(job_id_or_name, workspace_id)
    if job_id is None:
      raise ValueError(f"No serving found with name {job_name}")

    logger.debug(f"Latching to job '{job_name}' ({job_id})")
    
    def forward(*args, **kwargs):
      """This is the forward method for a NBX-Job. All the parameters will be passed through Relics."""
      logger.debug(f"Running job '{job_name}' ({job_id})")
      job = Job(job_id, workspace_id)
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
    _op._raise_on_io = False
    _op.forward = forward
    _op._op_type = OperatorType.JOB

    return _op

  @classmethod
  def from_serving(cls, deployment_id_or_name, token: str, workspace_id: str):
    """Latch to an existing serving operator

    Args:
      deployment_id_or_name (str): The id or name of the deployment
      token (str): The token to access the deployment, get it from settings
      workspace_id (str, optional): The workspace id. Defaults to None.
    """
    if not workspace_id:
      raise DeprecationWarning("Personal workspace does not support serving")
    serving_id, serving_name = _get_deployment_data(deployment_id_or_name, workspace_id)
    if serving_id is None:
      raise ValueError(f"No serving found with name {serving_name}")
    
    logger.debug(f"Latching to serving '{serving_name}' ({serving_id})")

    # common things
    session = requests.Session()
    session.headers.update({"NBX-KEY": token})

    REQ = type("REQ", (object,), {})

    # get the OpenAPI spec for the serving
    url = f"https://api.nimblebox.ai/{serving_id}/openapi.json"
    r = session.get(url)
    data = r.json()

    # get the arguments
    data = data["components"]["schemas"]
    key = list(filter(lambda x: x.endswith("_Request"), data))[0]
    comp = data[key]
    args = comp["properties"]
    args_dict = {k: v.get("default", REQ) for k, v in args.items()}

    # define the forward function for this serving operator, the objective is that this will be able to handle
    # args, kwargs just like how it works on the local machine
    def forward(*args, **kwargs):
      # check in the kwargs if we have any arguments to pass
      _data = {} # create the final dicst 
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

      # make a POST call to /forward and return the json
      url = f"https://api.nimblebox.ai/{serving_id}/forward"
      logger.debug(f"Hitting URL: {url}")
      logger.debug(f"Data: {kwargs}")
      r = session.post(url, json = _data)
      if r.status_code != 200:
        raise Exception(r.text)
      try:
        data = r.json()
      except:
        data = {}
        print(r.text)
      return data

    # create the class and override some values to make more sense
    _op = cls()
    _op.__qualname__ = "serving_" + key[:-8]
    _op._raise_on_io = False
    _op.forward = forward
    _op._op_type = OperatorType.SERVING
    return _op

  @classmethod
  def fn(cls):
    """Wraps the function as an Operator, so you can use all the same methods as Operator"""
    op = cls()
    def wrap(fn):
      op.forward = fn # override the forward function
      op.__doc__ = fn.__doc__ # override the docstring
      op.__qualname__ = "fn_" + fn.__name__ # override the name

      # this is necessary for getting the remote operations running
      op.__file__ = inspect.getfile(fn) # override the file
      op._op_type = OperatorType.WRAP
      return op
    return wrap

  # /mixin

  # information passing/

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

  def __setattr__(self, key, value: 'Operator'):
    obj = getattr(self, key, None)
    if (
      key != "forward" and
      obj is not None and
      callable(obj) and
      not isinstance(value, Operator)
    ):
      raise AttributeError(f"cannot assign {key} as it is already a method")
    if isinstance(value, Operator):
      if not "_operators" in self.__dict__:
        raise AttributeError("cannot assign operator before Operator.__init__() call")
      if key in self.__dict__ and key not in self._operators:
        raise KeyError(f"attribute '{key}' already exists")
      self._operators[key] = value

      # map the operator to the current resource
      self._op_to_resource_map[key] = self._current_resource
    self.__dict__[key] = value

  def propagate(self, **kwargs):
    """Set kwargs for each child in the Operator"""
    for k, v in kwargs.items():
      setattr(self, k, v)
    for c in self.children:
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

  def operators(self):
    r"""Returns an iterator over all operators in the job."""
    for _, module in self.named_operators():
      yield module

  def named_operators(self, memo = None, prefix: str = '', remove_duplicate: bool = True):
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
        for m in module.named_operators(memo, submodule_prefix, remove_duplicate):
          yield m

  @property
  def children(self) -> Iterable['Operator']:
    """Get children of the operator."""
    return self._operators.values()

  @property
  def inputs(self):
    if self._raise_on_io:
      args = inspect.getfullargspec(self.forward).args
      try:
        args.remove('self')
      except:
        raise ValueError("forward function must have 'self' as first argument")
      if self._inputs:
        args += self._inputs
      return args
    return []

  def __call__(self, *args, **kwargs):
    # # Type Checking and create input dicts
    # inputs = self.inputs
    # if self._raise_on_io:
    #   len_inputs = len(args) + len(kwargs)
    #   if len_inputs > len(inputs):
    #     raise ValueError(f"Number of arguments ({len(inputs)}) does not match number of inputs ({len_inputs})")
    #   elif len_inputs < len(args):
    #     raise ValueError(f"Need at least arguments ({len(args)}) but got ({len_inputs})")

    input_dict = {}
    # for i, arg in enumerate(args):
    #   input_dict[self.inputs[i]] = arg
    # for key, value in kwargs.items():
    #   if key in inputs:
    #     input_dict[key] = value

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
    # if out is None:
    #   outputs = {"out_0": str(type(None))}
    # elif isinstance(out, dict):
    #   outputs = {k: str(type(v)) for k, v in out.items()}
    # elif isinstance(out, (list, tuple)):
    #   outputs = {f"out_{i}": str(type(v)) for i, v in enumerate(out)}
    # else:
    #   outputs = {"out_0": str(type(out))}

    logger.debug(f"Ending operator '{self.__class__.__name__}': {self.node.id}")
    _ts = get_current_timestamp()
    self.node.run_status.MergeFrom(RunStatus(end = _ts, outputs = outputs,))
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
      if cls_item and cls_item.__class__.__base__ == Operator:
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
    deployment_type = "job",
    resource: Resource = None,
    *,
    _unittest = False
  ) -> 'Operator':
    """Uploads relevant files to the cloud and deploys as a batch process or and API endpoint, returns the relevant
    ``.from_job()`` or ``.from_serving`` Operator. This uploads the entire folder where the caller file is located.
    In which case having a ``.nboxignore`` and ``requirements.txt`` will also be moved over.
    
    Args:
      workspace_id (str): The workspace id to deploy to.
      id_or_name (str, optional): The id or name of the deployment. Defaults to None.
      deployment_type (str, optional): The type of deployment. Defaults to "job".
      resource (Resource, optional): The resource to deploy to. Defaults to None.
    """
    if self._op_type == OperatorType.UNSET:
      fp = inspect.getfile(self.__class__)
      name = self.__qualname__
    elif self._op_type in [OperatorType.JOB, OperatorType.SERVING]:
      raise ValueError("Cannot deploy an operator that is already deployed")
    elif self._op_type == OperatorType.WRAP:
      fp = self.__file__
      name = self.__qualname__[3:] # to account for "fn_"

    logger.info(f"Deploying '{name}' from '{fp}'")
    logger.info(f"Deployment Type: {deployment_type}")

    # so basically till now all we know is that `from fp import name` and `name()`. This process can now
    # be used to create a custom `nbx_user.py` file from which `exe.py` can import the three functions
    # and use them the conventional way.

    # create a temporary directory to store all the files
    nbx_folder = os.path.join(env.NBOX_HOME_DIR(), ".auto_dep", name)
    new_folder(nbx_folder)

    # just create a resource.pb, if it's empty protobuf will work it out
    rsp = os.path.join(nbx_folder, "resource.pb")
    write_binary_to_file(resource or DEFAULT_RESOURCE, file = rsp)

    # copy over all the files and wait for it, else the changes below won't be reflected
    folder, file = os.path.split(fp)
    print(folder, file)
    Popen(f'cp -r {folder}/ {nbx_folder}/', shell = True).wait()

    # in a way nbx_user.py and the three functions could simply be a router and the real functions can be
    # imported from the orginal file.
    with open(os.path.join(nbx_folder, "nbx_user.py"), "w") as f:
      f.write(f'''# Autogenerated for .deploy() call
from nbox.messages import read_file_to_binary
from nbox.hyperloop.job_pb2 import Resource

from {file.split('.')[0]} import {name}

def get_op(*a):
  out = {name}{'() # initialise the operator' if self._op_type == OperatorType.UNSET else f' # no need to initiliaze since is wrapped'}
  return out

get_resource = lambda: read_file_to_binary('{rsp}', message = Resource())
get_schedule = lambda: None
''')

    if _unittest:
      return

    # now we have created the entire folder and we can run it
    fn = Job.upload if deployment_type == "job" else Serve.upload
    out: Union[Job, Serve] = fn(
      init_folder = nbx_folder,
      id_or_name = id_or_name or "dot_deploy_runs", # all the .deploy() methods will be called dot_deploy_runs
      workspace_id = workspace_id,
    )

    return self.from_job(
      job_id_or_name = out.id,
      workspace_id = workspace_id,
    )

  def parallel(self):
    pass
  # /nbx

operator = Operator.fn

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
