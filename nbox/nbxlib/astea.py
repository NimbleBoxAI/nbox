# this file contains code for our AST engine that parses given code, indexes it and then can be queried
# to get things. It's nothing as fancy as SCP, but it will need to have better UX than on_functions.
# Here's a list of usecases till now:
# - find a given object (str) in the given file and return all the relevant information about it

from functools import lru_cache
import os
import re
import ast as A
from enum import Enum
from typing import List, Set, Union

class IndexTypes(Enum):
  FUNCTION = A.FunctionDef
  CLASS = A.ClassDef
  VARIABLE = A.Name
  IMPORT = A.Import
  IMPORT_FROM = A.ImportFrom
  ASSIGN = A.Assign
  EXPRESSION = A.Expr
  IF = A.If
  FILE = "file"

  def all():
    return [
      IndexTypes.FUNCTION, IndexTypes.CLASS, IndexTypes.VARIABLE, IndexTypes.IMPORT, IndexTypes.IMPORT_FROM,
      IndexTypes.ASSIGN, IndexTypes.EXPRESSION, IndexTypes.IF, IndexTypes.FILE
    ]

class Astea:
  def __init__(
    self,
    fname: str = "",
    *,
    name: str = "",
    type: IndexTypes = None,
    node: A.AST = None,
    code_lines: List[str] = None,
    order_index: int = -1,
  ) -> None:
    """This is an AST node, that can be called upon itself to traverse in a natural human way. Only `fname` is an important
    argument, others will be populated automatically.
    Args:
      fname (str): the file name to parse
    """

    if fname:
      assert os.path.exists(fname), f"file/folder {fname} does not exist"
      if os.path.isdir(fname):
        self._fname = os.path.join(fname, "__init__.py")
      type = IndexTypes.FILE
      with open(fname, 'r') as f:
        code = f.read()
        node: A.Module = A.parse(code)
    elif name:
      pass
    else:
      raise ValueError("Must provide either fname or name")
    
    self.name = name
    self.type = type
    self.node = node
    self.code_lines = code_lines
    self.order_index = order_index

  def __repr__(self) -> str:
    return f"{self.type.name} {self.name}"

  def __hash__(self) -> int:
    return repr(self).__hash__()

  @property
  @lru_cache(1)
  def index(self) -> List['Astea']:
    items = set()
    if hasattr(self.node, 'body'):
      for i, n in enumerate(self.node.body):
        if type(n) == A.FunctionDef:
          tea = Astea(name=n.name, type=IndexTypes.FUNCTION, node=n, code_lines=self.code_lines, order_index=i)
          items.add(tea)
        elif type(n) == A.ClassDef:
          tea = Astea(name=n.name, type=IndexTypes.CLASS, node=n, code_lines=self.code_lines, order_index=i)
          items.add(tea)
        elif type(n) == A.Import:
          for j, nn in enumerate(n.names):
            tea = Astea(name=nn.name, type=IndexTypes.IMPORT, node=n, code_lines=self.code_lines, order_index=i+j)
            items.add(tea)
        elif type(n) == A.ImportFrom:
          for j, nn in enumerate(n.names):
            tea = Astea(name=nn.name, type=IndexTypes.IMPORT_FROM, node=n, code_lines=self.code_lines, order_index=i+j)
            items.add(tea)
        elif type(n) == A.Expr:
          tea = Astea(name=n.value, type=IndexTypes.EXPRESSION, node=n, code_lines=self.code_lines, order_index=i)
          items.add(tea)
        elif type(n) == A.Assign:
          for j, nn in enumerate(n.targets):
            if hasattr(n, "id"):
              tea = Astea(name=nn.id, type=IndexTypes.VARIABLE, node=n, code_lines=self.code_lines, order_index=i+j)
            else:
              pass
            items.add(tea)
        else:
          continue
    items = sorted(list(items), key=lambda x: x.order_index)
    return items

  def find(self, x: str, types: Union[IndexTypes, List[IndexTypes]] = None) -> list:
    """Find all the instances of x in the current node, and return a list of IndexItems.
    
    - if there is a '.' (dot) in `x` find will try to perform a recursive search, you can escape the dot with a backslash
    """
    if types is None:
      types = IndexTypes.all()
    elif isinstance(types, IndexTypes):
      types = [types]

    # get the sections of the search string
    sections = []
    for s in re.split(r'(?<!\\)\.', x):
      sections.append(s.replace(r'\.', '.'))

    res = []
    node = self
    for i, s in enumerate(sections):
      if i == len(sections) - 1:
        # last section, we need to find the actual item
        for item in node.index:
          if item.name == s and item.type in types:
            res.append(item)
      else:
        # we need to find the next node
        for item in node.index:
          if item.name == s and item.type in types:
            node = item
            break
        else:
          # we didn't find the node, so we can't continue
          break
    return res

  def filter(self, types: Union[IndexTypes, List[IndexTypes]] = None, r: str = "") -> list:
    """Filter the index by the given types"""
    if types is None:
      types = IndexTypes.all()
    elif isinstance(types, IndexTypes):
      types = [types]
    items = [item for item in self.index if item.type in types]
    if r:
      items = [item for item in items if re.search(r, item.name)]
    return items
