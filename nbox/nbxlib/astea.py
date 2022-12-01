import os
import re
import ast as A
from enum import Enum
from functools import lru_cache
from typing import List, Set, Union
from hashlib import sha256 as _hash

def sha256(x: str):
  return _hash(x.encode()).hexdigest()

class IndexTypes(Enum):
  """
  This is our list of index types, these can be used to search through things in the AST. Here's some description on
  what each of these mean. There are a few special ones for convinience:
  - FILE: which means that the current location is the file itself, also can be considered as ast.Module
  - DOCSTRING: which is a special type of expression that is a docstring, is defined to use a different parsing logic
  """
  FUNCTION = A.FunctionDef
  CLASS = A.ClassDef
  VARIABLE = A.Name
  IMPORT = A.Import
  IMPORT_FROM = A.ImportFrom
  ASSIGN = A.Assign
  EXPRESSION = A.Expr
  IF = A.If
  FILE = "file"
  DOCSTRING = "__doc__"

  def all():
    """returns all the valid index types"""
    return [
      IndexTypes.FUNCTION,
      IndexTypes.CLASS,
      IndexTypes.VARIABLE,
      IndexTypes.IMPORT,
      IndexTypes.IMPORT_FROM,
      IndexTypes.ASSIGN,
      IndexTypes.EXPRESSION,
      IndexTypes.IF,
      IndexTypes.FILE
    ]


def get_code_section(node: A.AST, code_lines: List[str]):
  """generic function to return the code section for a node"""
  if type(code_lines) == str:
    code_lines = code_lines.splitlines()
  else:
    assert type(code_lines) == list and type(code_lines[0]) == str
  sl, so, el, eo = node.lineno, node.col_offset, node.end_lineno, node.end_col_offset
  code = ""
  if sl == el:
    code = code_lines[sl-1][so:eo]
  else:
    for i in range(sl - 1, el, 1):
      if i == sl - 1:
        code += code_lines[i][so:]
      elif i == el - 1:
        code += "\n" + code_lines[i][:eo]
      else:
        code += "\n" + code_lines[i]
  return code



class Astea:
  """### Engineer's Note
  
  Astea is designed from ground up to suit needs of code traversal and search. It's not a full AST engine, but it's
  good enough for our needs. In the first version, we wrote `nbox.framework.on_functions` which was written to generate
  DAGs for NBX-JobsFlow, it was overly complicated and not well engineered for great UX. So we ended up writing
  something brand new which reflects more like how humans work traverse code. This is very important because down the
  road we want to do really insane and powerful things with code.
  """
  def __init__(
    self,
    fname: str = "",
    code: str = "",
    *,
    name: str = "",
    type: IndexTypes = None,
    node: A.AST = None,
    code_lines: List[str] = None,
    order_index: int = -1,
  ) -> None:
    """This is an AST node, that can be called upon itself to traverse in a natural human way. Only `fname` is an important
    argument, others will be populated automatically.

    Examples:
      # Creating a simple tea:
      >>> from nbox.nbxlib import astea as A
      >>> tea = A.Astea(fname = A.__file__)

    Args:
      fname (str): the file name to parse. If this is not provided, then all other values must be provided.
      code (str): the code text can be provided instead of the 
      name (str, optional): The name of the node. Defaults to "".
      type (IndexTypes, optional): The type of the node. Defaults to None.
      node (A.AST, optional): The AST node. Defaults to None.
      code_lines (List[str], optional): The code lines. Defaults to None.
      order_index (int, optional): The order index of the node. Defaults to -1.
    """

    if fname:
      assert os.path.exists(fname), f"file/folder {fname} does not exist"
      if os.path.isdir(fname):
        fname = os.path.join(fname, "__init__.py")
      type = IndexTypes.FILE
      with open(fname, 'r') as f:
        code = f.read()
        node: A.Module = A.parse(code)
      code_lines = code.splitlines()
      name = fname
    elif code:
      type = IndexTypes.FILE
      node: A.Module = A.parse(code)
      code_lines = code.splitlines()
      name = "0"
    elif name:
      pass
    else:
      raise ValueError("Must provide either fname/code or name")

    self.name = name
    self.type = type
    self.node = node
    self.code_lines = code_lines
    self.order_index = order_index

    if hasattr(node, "lineno"):
      self._code = get_code_section(node, code_lines)
    else:
      self._code = "\n".join(self.code_lines)
    self._sha256 = sha256(self._code)

  def __repr__(self) -> str:
    return f"{self.type.name} {self.name} {self._sha256[:6]}"

  def __hash__(self) -> int:
    # since we know that eventually the sha256 of the code is the only unique identifier for nay piece of text
    # we can return the first 16 bytes of the sha256 as the hash, this is also useful since same object in two
    # different python processes will have different hashes, but the sha256 will always be the same.
    return int(self._sha256[:16], 16)

  @lru_cache()
  def docstring(self) -> str:
    """Get the docstring of the current node, is valid only for functions and classes. This will automatically
    perform relevant dedentation and return the docstring as a single string."""
    docstring = ""
    if type(self.node) in [A.FunctionDef, A.ClassDef, A.Module]:
      # check if this has a documentation string
      if self.node.body and isinstance(self.node.body[0], A.Expr) and isinstance(self.node.body[0].value, A.Str):
        docstring = self.node.body[0].value.s
    else:
      raise ValueError(f"Can't get docstring for {self.name} ({self.type})")
    docstring = docstring.strip()

    # in some cases we need to remove the indentation from the docstring
    doc_lines = docstring.splitlines()
    if len(doc_lines) > 1:
      _l = doc_lines[1]
      _i = 1
      while not _l and _i < len(doc_lines):
        _i += 1
        _l = doc_lines[_i]
      indent = len(_l) - len(_l.lstrip())
      docstring = doc_lines[0] + "\n" + "\n".join([line[indent:] for line in doc_lines[1:]])
    return docstring

  @property
  @lru_cache(1)
  def index(self) -> List['Astea']:
    """Returns a list of all the AST nodes (same `Astea`) in the current node. This is a recursive function and user/bot
    is expected to go over this to traverse."""
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
        
        # if this node is a class and node 'n' is a staticmethod then assign is considered
        elif type(n) == A.Assign:
          target = n.targets[0]
          if type(target) == A.Tuple:
            # iterate over the tuple and add each item as VARIABLE
            for j, nn in enumerate(target.elts):
              tea = Astea(name=nn.id, type=IndexTypes.VARIABLE, node=n, code_lines=self.code_lines, order_index=i+j)
              items.add(tea)
            continue
          elif type(target) == A.Name:
            tea = Astea(name=target.id, type=IndexTypes.VARIABLE, node=n, code_lines=self.code_lines, order_index=i)
            items.add(tea)
          else:
            raise ValueError(f"Unknown type {type(target)} for target {target}")
          if (type(self.node) == A.ClassDef or type(self.node) == A.Module):
            try:
              # this .id is very painful
              tea = Astea(name=n.targets[0].id, type=IndexTypes.ASSIGN, node=n, code_lines=self.code_lines, order_index=i)
              items.add(tea)
            except:
              # though we eventually want to get away with this try/except, we will keep it just in case
              pass
        else:
          continue
    items = sorted(list(items), key=lambda x: x.order_index)
    return items

  def find(self, x: str = "", types: Union[IndexTypes, List[IndexTypes]] = None) -> List['Astea']:
    """Find all the instances of x in the current node, and return a list of IndexItems.
    
    - if there is a '.' (dot) in `x`, `find` will try to perform a recursive search, you can escape the dot with a backslash
    - This currently does not work correctly when the same name is used in different types
    - if both `x` and `types` is not provided, then this behaves just like `.index`

    Examples:

      # Create a tea:
      >>> from nbox.nbxlib import astea as A
      >>> tea = A.Astea(fname = A.__file__)

      # Find things inside the tea:
      >>> cls_obj = tea.find("Astea")[0]      # search by name 
      >>> fn_obj = cls_obj.find('find')       # search inside any Astea
      >>> fn_obj = tea.find("Astea.find")[0]  # search inside things by using . (dot)

    Args:
      x (str): If provided, will find the name of the item to search for
      types (Union[IndexTypes, List[IndexTypes]], optional): The type of items to search for. Defaults to None.

    Returns:
      List['Astea']: A list of IndexItems that match the search
    """
    if types is None:
      types = IndexTypes.all()
    elif isinstance(types, IndexTypes):
      types = [types]

    # get the sections of the search string
    sections = []
    if x:
      for s in re.split(r'(?<!\\)\.', x):
        sections.append(s.replace(r'\.', '.'))

    res = []
    node = self
    if sections:
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
    else:
      # no sections, so we need to return all the items in the current node
      for item in node.index:
        if item.type in types:
          res.append(item)
    return res
