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
    if possible optimise the logic.
"""
# Some parts of the code are based on the pytorch nn.Module class
# pytorch license: https://github.com/pytorch/pytorch/blob/master/LICENSE
# modifications: research@nimblebox.ai 2022

import os
import zipfile
from hashlib import sha256
from tempfile import gettempdir
from collections import OrderedDict
from typing import Callable, Iterable, List
from google.protobuf.timestamp_pb2 import Timestamp

from . import utils as U
from .utils import logger
from .init import nbox_ws_v1
from .network import deploy_job, Schedule
from .framework.on_functions import get_nbx_flow
from .framework import AirflowMixin, PrefectMixin, LuigiMixin
from .hyperloop.job_pb2 import NBXAuthInfo, Job as JobProto, Resource
from .hyperloop.dag_pb2 import DAG, Node, RunStatus
from .nbxlib.tracer import Tracer

class Operator():
  _version: int = 1 # always try to keep this an i32

  def __init__(self) -> None:
    """Create an operator, which abstracts your code into sharable, bulding blocks which
    can then deployed on either NBX-Jobs or NBX-Deploy.
    
    Usage:

    .. code-block:: python
      
      class MyOperator(Operator):
        def __init__(self, ...):
          ... # initialisation of job happens here

          # use prebuilt operators to define the entire process
          from .nbxlib.ops import Shell
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
    """
    self._operators = OrderedDict() # {name: operator}
    self._op_trace = []
    self._tracer: Tracer = None

  # mixin/

  to_airflow_operator = AirflowMixin.to_airflow_operator
  to_airflow_dag = AirflowMixin.to_airflow_dag
  from_airflow_operator = classmethod(AirflowMixin.from_airflow_operator)
  from_airflow_dag = classmethod(AirflowMixin.from_airflow_dag)
  to_prefect_task = PrefectMixin.to_prefect_task
  to_prefect_flow = PrefectMixin.to_prefect_flow
  from_prefect_task = classmethod(PrefectMixin.from_prefect_task)
  from_prefect_flow = classmethod(PrefectMixin.from_prefect_flow)
  to_luigi_task = LuigiMixin.to_luigi_task
  to_luigi_flow = LuigiMixin.to_luigi_flow
  from_luigi_task = classmethod(LuigiMixin.from_luigi_task)
  from_luigi_flow = classmethod(LuigiMixin.from_luigi_flow)

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

  def propagate(self, **kwargs):
    """Set kwargs for each child in the Operator"""
    for c in self.children:
      c.propagate(**kwargs)
    for k, v in kwargs.items():
      setattr(self, k, v)

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
  
  # /information passing

  # properties/

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

  # user can manually define inputs if they want, avoid this user if you don't know how this system works

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

  # /properties

  def forward(self):
    raise NotImplementedError("User must implement forward()")

  def _register_forward(self, python_callable: Callable):
    """convienience method to register a forward method"""
    self.forward = python_callable

  # define these here, it's okay to waste some memory
  node = Node()
  source_edges: List[Node] = None

  def __call__(self, *args, **kwargs):
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

    logger.debug(f"Calling operator: {self.__class__.__name__}: {self.node.id}")
    _ts = Timestamp(); _ts.GetCurrentTime()
    self.node.run_status.CopyFrom(RunStatus(start = _ts, inputs = {k: str(type(v)) for k, v in input_dict.items()}))
    if self._tracer != None:
      self._tracer(self.node)

    # ---- USER SEPERATION BOUNDARY ---- #

    out = self.forward(**input_dict)

    # ---- USER SEPERATION BOUNDARY ---- #
    outputs = {}
    if out == None:
      outputs = {"out_0": str(type(None))}
    elif isinstance(out, dict):
      outputs = {k: str(type(v)) for k, v in out.items()}
    elif isinstance(out, (list, tuple)):
      outputs = {f"out_{i}": str(type(v)) for i, v in enumerate(out)}
    else:
      outputs = {"out_0": str(type(out))}

    logger.debug(f"Ending operator: {self.__class__.__name__}: {self.node.id}")
    _ts = Timestamp(); _ts.GetCurrentTime()
    self.node.run_status.MergeFrom(RunStatus(end = _ts, outputs = outputs,))
    if self._tracer != None:
      self._tracer(self.node)

    return out

  # nbx/

  def deploy(
    self,
    init_folder: str,
    job_id_or_name: str,
    workspace_id: str = None,
    schedule: Schedule = None,
    cache_dir: str = None,
    *,
    _unittest = False
  ):
    """Deploy this job on NBX-Jobs.

    Args:
        init_folder (str, optional): Name the folder to zip
        job_id_or_name (Union[str, int], optional): Name or ID of the job
        workspace_id (str): Workspace ID to deploy to, if not specified, will use the personal workspace
        schedule (Schedule, optional): If ``None`` will run only once, else will schedule the job
        cache_dir (str, optional): Folder where to put the zipped file, if ``None`` will be ``tempdir``
    Returns:
        Job: Job object
    """
    if workspace_id == None:
      stub_all_jobs = nbox_ws_v1.user.jobs
    else:
      stub_all_jobs = nbox_ws_v1.workspace.u(workspace_id).jobs

    jobs = list(filter(lambda x: x["job_id"] == job_id_or_name, stub_all_jobs()["data"]))
    if len(jobs) == 0:
      logger.info(f"No Job found with ID or name: {job_id_or_name}, will create a new one")
      job_name =  job_id_or_name
      job_id = None
    elif len(jobs) > 1:
      raise ValueError(f"Multiple jobs found for '{job_id_or_name}'")
    else:
      data = jobs[0]
      job_name = data["name"]
      job_id = data["job_id"]
    
    logger.info(f"Deploying job: {job_name}")
    logger.info(f"Job ID: {job_id}")

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
    dag: DAG = get_nbx_flow(self.forward)
    for _id, node in dag.flowchart.nodes.items():
      name = node.name
      if name.startswith("self."):
        name = name[5:]
      operator_name = "CodeBlock" # default
      cls_item = getattr(self, name, None)
      if cls_item and cls_item.__class__.__base__ == Operator:
        operator_name = cls_item.__class__.__name__
      node.operator = operator_name

    if schedule != None:
      logger.debug(f"Schedule: {schedule.get_dict}")

    _starts = Timestamp(); _starts.GetCurrentTime()
    job_proto = JobProto(
      id = job_id,
      name = job_name if job_name else U.get_random_name(True).split("-")[0],
      created_at = _starts,
      resource = Resource(),
      auth_info = NBXAuthInfo(workspace_id = workspace_id,),
      schedule = schedule.get_message() if schedule != None else None,
      dag = dag,
    )

    with open(U.join(init_folder, "job_proto.msg"), "wb") as f:
      f.write(job_proto.SerializeToString())

    # zip all the files folder
    all_f = U.get_files_in_folder(init_folder)
    all_f = [f[len(init_folder)+1:] for f in all_f] # remove the init_folder from zip

    for f in all_f:
      hash_ = sha256()
      with open(f, "rb") as f:
        for c in iter(lambda: f.read(2 ** 20), b""):
          hash_.update(c)
    hash_ = hash_.hexdigest()
    logger.info(f"SHA256 ( {init_folder} ): {hash_}")

    zip_path = U.join(cache_dir if cache_dir else gettempdir(), f"project-{hash_}.nbox")
    if not os.path.exists(zip_path):
      logger.info(f"Packing project to '{zip_path}'")
      with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for f in all_f:
          zip_file.write(f)
    else:
      logger.info(f"Found existing project zip file, will not re-pack: '{zip_path}'")

    if _unittest:
      return job_proto

    return deploy_job(zip_path = zip_path, job_proto = job_proto)

  # /nbx
