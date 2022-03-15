# weird file name, yessir. Why? because
# from nbox.framework.on_functions import PureFunctionParser
# read as "from nbox's framework on Functions import the pure-function Parser"

import ast
import base64
import inspect
from typing import Union
from uuid import uuid4

from ..utils import logger

# ==================

# dataclasses are not that good: these classes are for the Op

class DBase:
  def __init__(self, **kwargs):
    for k, v in kwargs.items():
      setattr(self, k, v)

  def get(self, k, v = None):
    return getattr(self, k, v)
  
  def get_dict(self):
    data = {}
    for k in self.__slots__:
      _obj = getattr(self, k, None)
      if _obj == None:
        continue
      if isinstance(_obj, DBase):
        data[k] = _obj.get_dict()
      elif _obj != None and isinstance(_obj, (list, tuple)) and len(_obj) and isinstance(_obj[0], DBase):
        data[k] = [_obj.get_dict() for _obj in _obj]
      else:
        data[k] = _obj
    return data

  def __repr__(self):
    return str(self.get_dict())

# ================== These classes are the code nodes

from ..hyperloop.dag_pb2 import DAG, Flowchart, Node, NodeInfo, Edge

# class ExpressionNodeInfo(DBase):
#   __slots__ = [
#     'name', # :str
#     'code', # :str (base64)
#     'nbox_string', # :str
#     'lineno', # :int
#     'col_offset', # :int
#     'end_lineno', # :int
#     'end_col_offset', # :int
#     'inputs', # :list[Dict[Any, Any]]
#     'outputs', # :list[str]
#   ]

# class IfNodeInfo(DBase):
#   __slots__ = [
#     'nbox_string', # :str
#     'conditions', # :list[ExpressionNodeInfo]
#     'inputs', # :list[Dict[Any, Any]]
#     'outputs', # :list[str]
#   ]

# class ForNodeInfo(DBase):
#   __slots__ = [
#     'nbox_string', # :str
#     'iterable', # :list[str]
#     'inputs', # :list[Dict[Any, Any]]
#     'info', # :ExpressionNodeInfo
#     'code', # :str
#   ]

# class ReturnNodeInfo(DBase):
#   __slots__ = [
#     'nbox_string', # :str
#   ]

# # ================== These classes create the DAG

# class RunStatus(DBase):
#   __slots__ = [
#     'start', # :str
#     'end', # :str
#     'inputs', # :Dict
#     'outputs', # :Dict
#   ]

# class Node(DBase):
#   __slots__ = [
#     'id', # :str
#     'execution_index', # :int
#     'name', # :str
#     'type', # :str: OneOf['op-node', 'if-node']
#     'info', # :Union[ExpressionNodeInfo, IfNodeInfo]
#     'operator', # :str
#     'nbox_string', # :str
#     'run_status', # :RunStatus
#   ]

# class Edge(DBase):
#   __slots__ = [
#     'id', # :str
#     'source', # :str
#     'target', # :str
#     'type', # :str
#     'nbox_string', # :str
#   ]


# ==================
# the next set of functions are meant as support methods to create nbox_strings IR.

class NboxStrings:
  OP_TO_STRING = {
    "function": "FUNCTION: {name} ( {inputs} ) => [ {outputs} ]",
    "define": "DEFINE: {name} ( {inputs} )",
    "for": "FOR: ( {iter} ) => [ {target} ]",
    "return": "RETURN: [ {value} ]",
  }

  def __init__(self):
    pass

  def function(self, name, inputs, outputs):
    return self.OP_TO_STRING["function"].format(
      name=name,
      inputs=", ".join([f"{x['kwarg']}={x['value']}" for x in inputs]),
      outputs=", ".join(outputs)
    )

  def define(self, name, inputs):
    return self.OP_TO_STRING["define"].format(
      name=name,
      inputs=", ".join([f"{x['kwarg']}={x['value']}" for x in inputs])
    )

  def for_loop(self, iter, target):
    return self.OP_TO_STRING["for"].format(
      iter=iter,
      target=", ".join(target)
    )

  def return_statement(self, value):
    return self.OP_TO_STRING["return"].format(
      value=value
    )

nbxl = NboxStrings()

def write_program(nodes):
  for i, n in enumerate(nodes):
    logger.debug(f"{i:03d}|{n.get('nbox_string', n.get('info').get('nbox_string'))}")


# ==================

def get_code_portion(cl, lineno, col_offset, end_lineno, end_col_offset, b64 = True, **_):
  sl, so, el, eo = lineno, col_offset, end_lineno, end_col_offset
  if sl == el:
    return cl[sl-1][so:eo]
  code = ""
  for i in range(sl - 1, el, 1):
    if i == sl - 1:
      code += cl[i][so:]
    elif i == el - 1:
      code += "\n" + cl[i][:eo]
    else:
      code += "\n" + cl[i]
  
  # convert to base64
  if b64:
    return base64.b64encode(code.encode()).decode()
  return code

def parse_args(node):
  inputs = []
  for a in node.args:
    a = a.arg if isinstance(a, ast.arg) else a
    inputs.append({
      "kwarg": None,
      "value": a,
    })
  for a in node.kwonlyargs:
    inputs.append({
      "kwarg": a[0],
      "value": a[1],
    })
  if node.vararg:
    inputs.append({
      "kwarg": "*"+node.vararg.arg,
      "value": None
    })
  if node.kwarg:
    inputs.append({
      "kwarg": "**"+node.kwarg.arg,
      "value": None
    })
  return inputs

def get_name(node):
  if isinstance(node, ast.Name):
    return node.id
  elif isinstance(node, ast.Attribute):
    return get_name(node.value) + "." + node.attr
  elif isinstance(node, ast.Call):
    return get_name(node.func)

def parse_kwargs(node, lines):
  if isinstance(node, ast.Name):
    return node.id
  elif isinstance(node, ast.Constant):
    val = node.value
    return val
  elif isinstance(node, ast.keyword):
    arg = node.arg
    value = node.value
    if 'id' in value.__dict__:
      # arg = my_model
      return (arg, value.id)
    elif 'value' in value.__dict__:
      if isinstance(value.value, ast.Call):
        # arg = my_model(...)
        return (arg, get_code_portion(lines, b64 = False, **value.value.__dict__))
      else:
        # arg = 20
        return (arg, value.value)
    elif 'func' in value.__dict__:
      #   arg = some_function(with, some=args)
      #   ^^^   ^^^^^
      # kwarg   value
      return {"kwarg": arg, "value": get_code_portion(lines, b64 = False, **value.__dict__)}
  elif isinstance(node, ast.Call):
    return get_code_portion(lines, **node.func.__dict__)

def node_assign_or_expr(node, lines, node_proto: Node) -> Union[Node, None]:
  # print(get_code_portion(lines, b64 = False, **node.__dict__))
  value = node.value
  try:
    name = get_name(value.func)
  except AttributeError:
    return None
  args = [parse_kwargs(x, lines) for x in value.args + value.keywords]
  inputs = []
  for a in args:
    if isinstance(a, dict):
      inputs.append(a)
      continue
    inputs.append({
      "kwarg": a[0] if isinstance(a, tuple) else None,
      "value": a[1] if isinstance(a, tuple) else a,
    })

  outputs = []
  if isinstance(node, ast.Assign):
    targets = node.targets[0]
    outputs = [parse_kwargs(x, lines) for x in targets.elts] \
      if isinstance(targets, ast.Tuple) \
      else [parse_kwargs(targets, lines)
    ]

  # return ExpressionNodeInfo(
  #   name = name,
  #   inputs = inputs,
  #   outputs = outputs,
  #   nbox_string = 
  #   code = get_code_portion(lines, **node.__dict__),
  #   lineno = node.lineno,
  #   col_offset = node.col_offset,
  #   end_lineno = node.end_lineno,
  #   end_col_offset = node.end_col_offset,
  # )
  node_proto.operator = Node.NodeTypes.LOOP

  # updates for NodeInfo
  node_proto.info.name = name
  node_proto.info.nbox_string = nbxl.function(name, inputs, outputs),


def node_if_expr(node, lines, node_proto: Node) -> Node:
  def get_conditions(node, lines, conds = []):
    if not hasattr(node, "test"):
      else_cond = list(filter(lambda x: x["condition"] == "else", conds))
      if not else_cond:
        conds.append({
          "condition": "else",
          "code": dict(
            lineno = node.lineno,
            col_offset = node.col_offset,
            end_lineno = node.end_lineno,
            end_col_offset = node.end_col_offset,
          )
        })
      else:
        cond = else_cond[0]
        cond["code"]["end_lineno"] = node.end_lineno
        cond["code"]["end_col_offset"] = node.end_col_offset
    else:
      condition = get_code_portion(lines, **node.test.__dict__)

      # need to run this last or "else" comes up first
      conds.append({
        "condition": condition,
        "code": {
          "lineno": node.lineno,
          "col_offset": node.col_offset,
          "end_lineno": node.end_lineno,
          "end_col_offset": node.end_col_offset,
        }
      })
      for x in node.orelse:
        get_conditions(x, lines, conds)

    return conds
  
  # get all the conditions and structure as ExpressionNodeInfo

  all_conditions = get_conditions(node, lines, conds = [])
  ends = []
  for b0, b1  in zip(all_conditions[:-1], all_conditions[1:]):
    ends.append([b0["code"], b1["code"]])
  for i in range(len(ends)):
    ends[i] = {
      "lineno": ends[i][0]["lineno"],
      "col_offset": ends[i][0]["col_offset"],
      "end_lineno": ends[i][1]["lineno"],
      "end_col_offset": ends[i][1]["col_offset"],
    }
  ends += [all_conditions[-1]["code"]]

  conditions = {}
  for i, c in enumerate(all_conditions):
    box = ends[i]
    _node = NodeInfo(
      name = f"if-{i}",
      nbox_string = c["condition"],
      lineno = box['lineno'],
      col_offset = box['col_offset'],
      end_lineno = box['end_lineno'],
      end_col_offset = box['end_col_offset'],
      inputs = [],
      outputs = [],
    )
    conditions[_node.name] = _node

  node_proto.info.nbox_string = "IF: { " + ", ".join(x.nbox_string for x in conditions) + " }"
  node_proto.info.conditions = conditions
  # return IfNodeInfo(
  #   conditions = conditions,
  #   nbox_string = nbox_string,
  #   inputs = [],
  #   outputs = []
  # )
  return node_proto

def node_for_expr(node, lines, node_proto: Node) -> Node:
  node_proto.info.nbox_string = nbxl.for_loop(get_code_portion(lines, **node.value.__dict__))
  node_proto.name = f"for-{i}"
  node_proto.operator = Node.NodeTypes.LOOP
  # targets = [x.id for x in node.target.elts]
  # iter_str = get_code_portion(lines, **node.iter.__dict__)
  # code = get_code_portion(lines, **node.body[0].__dict__)
  # return ForNodeInfo(
  #   targets = targets,
  #   iter_str = iter_str,
  #   code = code,
  #   nbox_string = nbxl.for_loop(iter_str, targets),
  # )
  return node_proto

def node_return(node, lines, node_proto: Node) -> Node:
  node_proto.info.nbox_string = nbxl.return_statement(get_code_portion(lines, **node.value.__dict__))
  node_proto.name = f"return"
  node_proto.operator = Node.NodeTypes.RETURN
  return node_proto

def def_func_or_class(node, lines):
  out = {
    "name": node.name,
    "code": get_code_portion(lines, **node.__dict__),
    "type": "def-node"
  }
  if isinstance(node, ast.FunctionDef):
    out.update({"func": True, "inputs": parse_args(node.args)})
  else:
    out.update({"func": False, "inputs": []})
  return out


# ==================

type_wise_logic = {
  # defns ------
  ast.FunctionDef: def_func_or_class,
  ast.ClassDef: def_func_or_class,

  # nodes ------
  ast.Assign: node_assign_or_expr,
  ast.Expr: node_assign_or_expr,
  ast.If: node_if_expr,
  ast.For: node_for_expr,

  # Return ------
  ast.Return: node_return,

  # todos ------
  # ast.AsyncFunctionDef: async_func_def,
  # ast.Await: node_assign_or_expr,
}

# ==================


def code_node(execution_index, expr, code_lines) -> Node:
  # code pieces that are not yet supported should still see the code
  return Node(
    id = str(uuid4()),
    execution_index = execution_index,
    name = f"codeblock-{execution_index}",
    type = "op-node",
    operator = "CodeBlock",
    nbox_string = f"CODE: {str(type(expr))}", # :str
    run_status = Node.RunStatus(), # no need tp initialise this object
    info = NodeInfo(
      name = f"codeblock-{execution_index}", # :str
      code = get_code_portion(code_lines, bs64 = True, **expr.__dict__), # :str (base64)
      nbox_string = None, # :str
      lineno = expr.lineno, # :int
      col_offset = expr.col_offset, # :int
      end_lineno = expr.end_lineno, # :int
      end_col_offset = expr.end_col_offset, # :int
      inputs = [], # :list[Dict[Any, Any]]
      outputs = [], # :list[str]
    )
  )


def get_nbx_flow(forward):
  """Get NBX flowchart. Read python grammar here:
  https://docs.python.org/3/reference/grammar.html

  Args:
      forward (callable): the function whose flowchart is to be generated
  """
  # get code string from operator
  code = inspect.getsource(forward).strip()
  code_lines = code.splitlines()
  node = ast.parse(code)

  edges = {} # this is the flow
  nodes = {} # this is the operators
  symbols_to_nodes = {} # this is things that are defined at runtime

  for i, expr in enumerate(node.body[0].body):
    # create the empty node that will be used everywhere
    if not type(expr) in type_wise_logic:
      node.info = code_node(i, expr, code_lines)
      nodes[node.id] = node
      continue

    output = type_wise_logic[type(expr)](expr, code_lines, node)
    if output is None:
      continue

    if "def" in output["type"]:
      symbols_to_nodes[output['name']] = {
        "info": output,
        "execution_index": i,
        "nbox_string": nbxl.define(output["name"], output["inputs"])
      }
      continue

    # if isinstance(output, ExpressionNodeInfo):
    
    # elif isinstance(output, IfNodeInfo):
    #   node_.name = f"if-{i}"
    #   node_.operator = Node.NodeTypes.BRANCHING
    #   node_.nbox_string = output.nbox_string
    
    # elif isinstance(output, ForNodeInfo):
      
    
    # elif isinstance(output, ReturnNodeInfo):
      

    # nodes[node_.id] = node_

  # edges for execution order can be added
  _node_ids = tuple(nodes.keys())
  for op0, op1 in zip(_node_ids[:-1], _node_ids[1:]):
    _id = f"edge-{op0}-X-{op1}"
    edges[_id] = Flowchart.Edge(
      id = _id,
      source = op0,
      target = op1,
      type = "execution-order",
      nbox_string = None
    )

  # from pprint import pprint
  # pprint(edges)
  # pprint(nodes)

  return DAG(
    flowchart = Flowchart(
      nodes = nodes, edges = edges
    ),
    symbols=symbols_to_nodes
  )
