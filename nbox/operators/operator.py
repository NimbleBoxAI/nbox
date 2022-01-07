# Some parts of the code are based on the pytorch nn.Module class
# pytorch license: https://github.com/pytorch/pytorch/blob/master/LICENSE
# due to requirements of stability, some type enforcing is performed

from collections import OrderedDict
from dataclasses import dataclass


@dataclass
class StateDictModel:
  state: str
  data: dict
  inputs: dict
  outputs: dict

  def __post_init__(self):
    self.data = OrderedDict(self.data)


class Operator:
  _version: int = 1

  def __init__(self) -> None:
    self._operators = OrderedDict() # {name: operator}

  # classmethods/

  def serialise(self):
    pass

  @classmethod
  def deserialise(cls, state_dict):
    pass

  @classmethod
  def from_airflow(cls, dag):
    pass

  def to_airflow(self, fpath = None):
    dag = None
    if fpath:
      # creates a file with the DAG at the given path with code for running the DAG
      pass
    return dag

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
    if obj is not None and callable(obj):
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
    r"""Returns an iterator over all operators in the job.

    Yields:
      Operator: a operator in the network

    Note:
      Duplicate modules are returned only once. In the following
      example, ``l`` will be returned only once.
    """
    for _, module in self.named_operators():
      yield module

  def named_operators(self, memo = None, prefix: str = '', remove_duplicate: bool = True):
    r"""Returns an iterator over all modules in the network, yielding
    both the name of the module as well as the module itself.

    Args:
        memo: a memo to store the set of modules already added to the result
        prefix: a prefix that will be added to the name of the module
        remove_duplicate: whether to remove the duplicated module instances in the result
        or not

    Yields:
        (string, Module): Tuple of name and module

    Note:
        Duplicate modules are returned only once. In the following
        example, ``l`` will be returned only once.
    """
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
  def inputs(self):
    import inspect
    args = inspect.getfullargspec(self.forward).args
    args.remove('self')
    return args

  @property
  def is_dag(self):
    graph = set()
    for idx, (name, op) in enumerate(self.named_operators()):
      name = "root" if name == "" else name
      if name in graph:
        return False
      graph.add(name)
    return True

  @property
  def state_dict(self) -> StateDictModel:
    return StateDictModel(
      state = self.__class__.__name__,
      data = {},
      inputs = self.inputs,
      outputs = self.outputs
    )

  # /properties

  def forward(self):
    raise NotImplementedError("User must implement forward()")

  def __call__(self, *args, **kwargs):
    assert self.is_dag

    # here we will add code for:
    # 1. (opt) type checking
    # 2. networking
    # 3. tracing

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

    # pass this through the user defined forward()
    return self.forward(**input_dict)

  # nbx/

  def deploy(group_name_or_id):
    pass

  # /nbx
