# this file contains code for our AST engine that parses given code, indexes it and then can be queried
# to get things. It's nothing as fancy as SCP, but it will need to have better UX than on_functions.
# Here's a list of usecases till now:
# - find a given object (str) in the given file and return all the relevant information about it

import ast as A
from enum import Enum
from typing import List, Union

class IndexTypes(Enum):
  FUNCTION = A.FunctionDef
  CLASS = A.ClassDef
  VARIABLE = A.Name
  IMPORT = A.Import
  IMPORT_FROM = A.ImportFrom

  def all():
    return [IndexTypes.FUNCTION, IndexTypes.CLASS, IndexTypes.VARIABLE, IndexTypes.IMPORT, IndexTypes.IMPORT_FROM]


class RowItem:
  def __init__(self, name: str, type: IndexTypes, node: A.AST) -> None:
    self.name = name
    self.type = type
    self.node = node


class Visiboi(A.NodeVisitor):
  def __init__(self) -> None:
    self.data = []

  def visit_FunctionDef(self, node: A.FunctionDef) -> None:
    self.data.append(RowItem(node.name, IndexTypes.FUNCTION, node))

  def visit_ClassDef(self, node: A.ClassDef) -> None:
    self.data.append(RowItem(node.name, IndexTypes.CLASS, node))

  def visit_Name(self, node: A.Name) -> None:
    self.data.append(RowItem(node.id, IndexTypes.VARIABLE, node))

  def visit_Import(self, node: A.Import) -> None:
    for n in node.names:
      self.data.append(RowItem(n.name, IndexTypes.IMPORT, node))

  def visit_ImportFrom(self, node: A.ImportFrom) -> None:
    for n in node.names:
      self.data.append(RowItem(n.name, IndexTypes.IMPORT_FROM, node))


class Astea:
  def __init__(self, fname: str) -> None:
    """This is our AST engine, and it works on file level."""
    self.fname = fname
    with open(fname, 'r') as f:
      self.code = f.read()
    self.node = A.parse(self.code)
    self.vis = Visiboi()
    self.vis.visit(self.node)
    self.index = self.vis.data

  def find(self, x: str, types: Union[IndexTypes, List[IndexTypes]] = None) -> list:
    """Find all the instances of x in the file, and return a list of RowItems."""
    if types is None:
      types = IndexTypes.all()
    elif isinstance(types, IndexTypes):
      types = [types]
    return [i for i in self.index if i.name == x and i.type in types]

