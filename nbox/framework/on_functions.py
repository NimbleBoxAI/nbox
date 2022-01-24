# weird file name, yessir. Why? because
# from nbox.framework.on_functions import PureFunctionParser
# read as "from nbox's framework on Functions import the pure-function Parser"

import ast
import inspect
from uuid import uuid4
from logging import getLogger
logger = getLogger()

# ==================

# dataclasses are not that good: these classes are for the Op

class DBase:
  def __init__(self, **kwargs):
    for k, v in kwargs.items():
      setattr(self, k, v)
  
  def get_dict(self):
    data = {}
    for k in self.__slots__:
      _obj = getattr(self, k)
      if isinstance(_obj, DBase):
        data[k] = _obj.get_dict()
      elif _obj and isinstance(_obj, list) and isinstance(_obj[0], DBase):
        data[k] = [_obj.get_dict() for _obj in _obj]
      else:
        data[k] = _obj
    return data

class ExpressionNodeInfo(DBase):
  __slots__ = [
    'name', # :str
    'code', # :str (base64)
    'nbox_string', # :str
    'lineno', # :int
    'col_offset', # :int
    'end_lineno', # :int
    'end_col_offset', # :int
    'inputs', # :list[Dict[Any, Any]]
    'outputs', # :list[str]
  ]

class IfNodeInfo(DBase):
  __slots__ = [
    'nbox_string', # :str
    'conditions', # :list[ExpressionNodeInfo]
    'inputs', # :list[Dict[Any, Any]]
    'outputs', # :list[str]
  ]


# these classes are for the FE
class RunStatus(DBase):
  __slots__ = [
    'start', # :str
    'end', # :str
    'inputs', # :Dict
    'outputs', # :Dict
  ]


class Node(DBase):
  __slots__ = [
    'id', # :str
    'execution_index', # :int
    'name', # :str
    'type', # :str
    'node_info', # :Union[ExpressionNodeInfo, IfNodeInfo]
    'operator', # :str
    'nbox_string', # :str
    'run_status', # :RunStatus
  ]

class Edge(DBase):
  __slots__ = [
    'id', # :str
    'source', # :str
    'target', # :str
    'type', # :str
    'nbox_string', # :str
  ]


# ==================
# the next set of functions are meant as support methods to create nbox_strings IR.

class NboxStrings:
  OP_TO_STRING = {
    "function": "FUNCTION: {name} ( {inputs} ) => [ {outputs} ]",
    "define": "DEFINE: {name} ( {inputs} )",
    "for": "FOR: {name} ( {iter} ) => ( {target} )",
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

  def _for(self, name, iter, target):
    return self.OP_TO_STRING["for"].format(
      name=name,
      iter=iter,
      target=target
    )

nbxl = NboxStrings()

def write_program(nodes):
  for i, n in enumerate(nodes):
    if n.nbox_string == None:
      print(f"{i:03d}|{n.node_info.nbox_string}")
    else:
      print(f"{i:03d}|{n.nbox_string}")

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
    import base64
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
  if isinstance(node, ast.Constant):
    val = node.value
    return val
  if isinstance(node, ast.keyword):
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
  if isinstance(node, ast.Call):
    return get_code_portion(lines, **node.func.__dict__)

def node_assign_or_expr(node, lines):
  # print(get_code_portion(lines, **node.__dict__))
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

  return ExpressionNodeInfo(
    name = name,
    inputs = inputs,
    outputs = outputs,
    nbox_string = nbxl.function(name, inputs, outputs),
    code = get_code_portion(lines, **node.__dict__),
    lineno = node.lineno,
    col_offset = node.col_offset,
    end_lineno = node.end_lineno,
    end_col_offset = node.end_col_offset,
  )

def node_if_expr(node, lines):
  def if_cond(node, lines, conds = []):
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
        if_cond(x, lines, conds)

    return conds
  
  # get all the conditions and structure as ExpressionNodeInfo
  
  all_conditions = if_cond(node, lines, conds = [])
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

  conditions = []
  for i, c in enumerate(all_conditions):
    box = ends[i]
    _node = ExpressionNodeInfo(
      name = f"if-{i}",
      nbox_string = c["condition"],
      code = get_code_portion(lines, **box),
      lineno = box['lineno'],
      col_offset = box['col_offset'],
      end_lineno = box['end_lineno'],
      end_col_offset = box['end_col_offset'],
      inputs = [],
      outputs = [],
    )
    conditions.append(_node)

  nbox_string = "IF: { " + ", ".join(x.nbox_string for x in conditions) + " }"
  return IfNodeInfo(
    conditions = conditions,
    nbox_string = nbox_string,
    inputs = [],
    outputs = []
  )

def def_func_or_class(node, lines):
  out = {"name": node.name, "code": get_code_portion(lines, **node.__dict__), "type": "def-node"}
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

  # todos ------
  # ast.AsyncFunctionDef: async_func_def,
  # ast.Await: node_assign_or_expr,
}

# ==================

def get_nbx_flow(forward):
  # get code string from operator
  code = inspect.getsource(forward).strip()
  code_lines = code.splitlines()
  node = ast.parse(code)

  edges = [] # this is the flow
  nodes = [] # this is the operators
  symbols_to_nodes = {} # this is things that are defined at runtime

  for i, expr in enumerate(node.body[0].body):
    # if isinstance(expr, ast.Module):
    #   continue

    if not type(expr) in type_wise_logic:
      # code pieces that are not yet supported should still see the code
      node = Node(
        id = str(uuid4()),
        execution_index = i,
        name = f"codeblock-{i}",
        type = "op-node",
        operator = "CodeBlock",
        node_info = ExpressionNodeInfo(
          name = f"codeblock-{i}", # :str
          code = get_code_portion(code_lines, bs64 = True, **expr.__dict__), # :str (base64)
          nbox_string = None, # :str
          lineno = expr.lineno, # :int
          col_offset = expr.col_offset, # :int
          end_lineno = expr.end_lineno, # :int
          end_col_offset = expr.end_col_offset, # :int
          inputs = [], # :list[Dict[Any, Any]]
          outputs = [], # :list[str]
        ),
      nbox_string = f"CODE: {str(type(expr))}", # :str
      run_status = RunStatus(start = None, end = None, inputs = [], outputs = [])
      )
      nodes.append(node)
      continue

    output = type_wise_logic[type(expr)](expr, code_lines)
    if output is None:
      continue

    if isinstance(output, ExpressionNodeInfo):
      output = Node(
        id = str(uuid4()),
        execution_index = i,
        name = output.name,
        type = "op-node",
        operator = "CodeBlock",
        node_info = output,
        nbox_string = None,
        run_status = RunStatus(start = None, end = None, inputs = [], outputs = [])
      )
      nodes.append(output)
    elif isinstance(output, IfNodeInfo):
      output = Node(
        id = str(uuid4()),
        execution_index = i,
        name = f"if-{i}",
        type = "op-node",
        operator = "Conditional",
        node_info = output,
        nbox_string = output.nbox_string,
        run_status = RunStatus(start = None, end = None, inputs = [], outputs = [])
      )
      nodes.append(output)
    elif "def" in output["type"]:
      symbols_to_nodes[output['name']] = {
        "node_info": output,
        "execution_index": i,
        "nbox_string": nbxl.define(output["name"], output["inputs"])
      }

  # edges for execution order can be added
  for op0, op1 in zip(nodes[:-1], nodes[1:]):
    edges.append(
      Edge(
        id = f"edge-{op0.id}-X-{op1.id}",
        source = op0.id,
        target = op1.id,
        type = "execution-order",
        nbox_string = None
    )
  )

  return {
    "flowchart": {
      "edges": [x.get_dict() for x in edges],
      "nodes": [x.get_dict() for x in nodes],
    },
    "symbols": symbols_to_nodes
  }
