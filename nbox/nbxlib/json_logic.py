import sys
from collections.abc import Iterable
from functools import reduce

def _if(*args):
  assert len(args) >= 3, f"Provide atleast (cond, _true, _false) syntax "
  assert len(args) % 2 == 1, f"There should be odd elements in the conditional, got {len(args)}"
  pairs = [args[i*2:i*2+2] for i, _ in enumerate(args[:-2:2])]
  pairs.append((True, args[-1]))
  for c,o in pairs:
    if c:
      return o

def _merge(*args):
  # takes a bunch of elements and translates them into a 1D Array
  # find and past solution from SO! don't use your brains on this
  array = []
  for a in args:
    if isinstance(a, Iterable) and not isinstance(a, str):
      for i in _merge(*a):
        array.append(i)
    else:
      array.append(a)
  return array


class JsonLogic():
  def __init__(self, rules, data = {}):
    """
    Modified for our needs from: https://github.com/nadirizr/json-logic-py
    This is a Python implementation of the following jsonLogic JS library: https://github.com/jwadhams/json-logic-js

    Args:
      rules (dict): The rules to be applied
      data (dict, optional): The data to be used in the rules. Defaults to None.
    """
    self.rules = rules
    self.data = data
    self.operations = {
      # accessing data
      "var" : (
        lambda a, not_found = None: reduce(
          lambda data, key: (
            data.get(key, not_found) if type(data) == dict else data[int(key)]
            if (type(data) in [list, tuple] and str(key).lstrip("-").isdigit())
            else not_found
          ),
          str(a).split("."),
        data
        )
      ),
      # "missing",
      # "missing_some",

      # logic and boolean operators
      "if": _if,
      "=="  : (lambda a, b: a == b),
      "===" : (lambda a, b: a is b),
      "!="  : (lambda a, b: a != b),
      "!==" : (lambda a, b: a is not b),
      "!"   : (lambda a: not a),
      # "!!",
      "or"  : (lambda *args: reduce(lambda total, arg: total or arg, args, False)),
      "and" : (lambda *args: reduce(lambda total, arg: total and arg, args, True)),

      # numeric operators
      ">"   : (lambda a, b: a > b),
      ">="  : (lambda a, b: a >= b),
      "<"   : (lambda a, b, c = None:  a < b if (c is None) else (a < b) and (b < c)),
      "<="  : (lambda a, b, c = None: a <= b if (c is None) else (a <= b) and (b <= c)),
      # between
      "min" : (lambda *args: min(args)),
      "max" : (lambda *args: max(args)),
      "+" : (lambda *args: reduce(lambda total, arg: total + float(arg), args, 0.0)),
      "*" : (lambda *args: reduce(lambda total, arg: total * float(arg), args, 1.0)),
      "-" : (lambda a, b=None: -a if b is None else a - b),
      "/" : (lambda a, b=None: a if b is None else float(a) / float(b)),
      "%"   : (lambda a, b: a % b),

      # Array opertors
      # "map",
      # "reduce",
      # "filter",
      # "all",
      # "none",
      # "some",
      "merge": _merge,
      "in"  : (lambda a, b: a in b if "__contains__" in dir(b) else False),

      # string operators
      # "in" defined above
      "cat" : (lambda *args: "".join(args)),
      "substr": (lambda a, b, c=None: a[b:c] if c else a[b:]),
      
      # Misc.
      "log" : (lambda a: a if sys.stdout.write(str(a)) else a),
      "count": (lambda *args: sum(1 if a else 0 for a in args)),
      "?:"  : (lambda a, b, c: b if a else c),
    }
  
  def apply(self):
    tests = self.rules
    data = self.data

    # You've recursed to a primitive, stop!
    if tests is None or type(tests) != dict:
      return tests

    op = tuple(tests.keys())[0]
    values = tests[op]
    
    if op not in self.operations:
      raise RuntimeError(f"Unrecognized operation '{op}'")

    # Easy syntax for unary operators, like {"var": "x"} instead of strict {"var": ["x"]}
    if type(values) not in [list, tuple]:
      values = [values]

    # Recursion!
    values = map(lambda val: JsonLogic(val, data).apply(), values)

    return self.operations[op](*values)
