"""
## NBX Operator Serving 

This file has all the functions on how the serving of an Operator works, it also acts as the default code that runs
on NBX Deploy service.

{% CallOut variant="success" label="If you find yourself using this reach out to NimbleBox support." /%}
"""
# https://github.com/apache/airflow/issues/7870


try:
  from pydantic import create_model
  import uvicorn

  from fastapi import FastAPI, APIRouter
  from fastapi.responses import JSONResponse, Response
  from fastapi.middleware.cors import CORSMiddleware
except ImportError:
  # if this is happening to you sir, why don't you come work with us?
  FastAPI = None

import json
import inspect
from typing import Any, Dict, Callable

from nbox.version import __version__
from nbox.operator import Operator
from nbox.nbxlib.operator_spec import OperatorType
from nbox.utils import py_from_bs64, py_to_bs64, logger

def serve_operator(
  op_or_app: Operator,
  host: str = "0.0.0.0",
  port: int = 8000,
  *,
  log_system_metrics: bool = True,
  log_user_io: bool = False,
  model_name: str = ""
):
  """
  Serve an operator or a FastAPI app on a given host and port.

  Args:
    op_or_app: The operator or FastAPI app to serve.
    host: The host to serve on.
    port: The port to serve on.
  """
  # TODO: @yashbonde server can behave like a proxy to a different running server so it can be flask app or anything
  # https://stackoverflow.com/questions/70610266/proxy-an-external-website-using-python-fast-api-not-supporting-query-params
  if FastAPI is None:
    logger.error("To run servers you will need to install the relevant dependencies:")
    logger.error("  pip install -U nbox[serving]")
    raise ImportError("fastapi not installed")

  app = FastAPI()
  app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
  )

  # define ping endpoint
  async def ping_fn():
    return {"message": "pong"}
  app.add_api_route("/", ping_fn, methods=["GET"], response_class=JSONResponse)

  # define metadata endpoint
  async def metadata():
    return {"metadata": {"name": model_name}}
  app.add_api_route("/metadata", metadata, methods=["GET"], response_class=JSONResponse)

  if type(op_or_app) == Operator:
    # a special route for Operators to communicate with each other
    async def who_are_you():
      return {"name": op_or_app.__qualname__, "nbox_version": __version__, "rest_api_style": "rest_{p}"}
    app.add_api_route("/who_are_you", who_are_you, methods=["GET"], response_class=JSONResponse)

    for route, fn in get_fastapi_routes(op_or_app):
      app.add_api_route(route, fn, methods=["POST"], response_class=JSONResponse)
  
  elif type(op_or_app) == FastAPI:
    app.mount("/x", op_or_app)

  uvicorn.run(app, host = host, port = port)


def get_fastapi_routes(op: Operator):
  """To keep seperation of responsibility the paths are scoped out like all the functions are
  in the `/method_{...}` and all the custom python code is in `/nbx_py_rpc`"""
  if op._op_type == OperatorType.WRAP_CLS:
    routes = []
    # add functions that the user has exposed
    wrap_class = op._op_spec.wrap_obj
    for p in dir(wrap_class.__class__):
      if p.startswith("_"):
        # we will forever ignore the functions that start with _ to provide like a private functions, just like how
        # python would disallow you from "__" functions, this networking layer does that for "_"
        continue
      fn = getattr(wrap_class, p)
      # TODO:@yashbonde >>> replace /method_{p} with /pyrpc/{p} and /method_{p}_rest with /rest/{p}
      routes.append((f"/method_{p}", get_fastapi_fn(fn)))
      routes.append((f"/rest_{p}", get_fastapi_fn(fn, _rest = True)))

    # add functions that the python itself can support
    routes.append((f"/nbx_py_rpc", nbx_py_rpc(op)))
  elif op._op_type in [OperatorType.JOB, OperatorType.SERVING]:
    raise RuntimeError("Cannot serve a job or serving operator")
  else:
    routes = [
      ("/forward", get_fastapi_fn(op.forward)),
      ("/forward_rest", get_fastapi_fn(op.forward, _rest = True)),
    ]
  return routes


# builder method is used to progrmatically generate api routes related information for the fastapi app
def get_fastapi_fn(fn, _rest = False) -> Callable:
  """This function is used to generate a fastapi route for a given function, it will take in a function
  and return a function that can be used as a fastapi route

  Args:
    fn (Callable): function to be used as a fastapi route
    _rest (bool, optional): if the function is a REST endpoint. Defaults to False.

  Returns:
    Callable: a function that can be used as a fastapi route
  """
  from pydantic import create_model

  # we use inspect signature instead of writing our own ast thing
  signature = inspect.signature(fn)
  data_dict = {}
  for param in signature.parameters.values():
    default = param.default
    annot = param.annotation
    if default == inspect._empty:
      default = None
    if param.annotation == inspect._empty:
      annot = Any
    data_dict[param.name] = (annot, default)

  # if your function takes in inputs then it is expected to be sent as query params, so create a pydantic
  # model and FastAPI will take care of the rest
  name = f"{fn.__name__}_Request"
  if _rest:
    name = f"{fn.__name__}_Rest_Request"
  base_model = create_model(name, **data_dict)
  base_model_rpc = create_model(name, **{k:(str, py_to_bs64(v[1])) for k,v in data_dict.items()})

  # pretty simple forward function, note that it gets operator using get_op which will be a cache hit
  async def generic_fwd(req: base_model_rpc, response: Response):
    # need to add serialisation to this function because user won't by default send in a serialised object
    data = req.dict()
    try:
      data = {k: py_from_bs64(v) for k, v in data.items()}
    except Exception as e:
      logger.error(f"Failed to convert {data} to python object")
      logger.error(e)
      response.status_code = 400
      return {"error": str(e)}

    try:
      out = fn(**data)
      return {"success": True, "value": py_to_bs64(out)}
    except Exception as e:
      response.status_code = 500
      return {"success": False, "message": str(e)}

  # pretty simple forward function for REST endpoints
  async def generic_fwd_rest(req: base_model, response: Response):
    # need to add serialisation to this function because user won't by default send in a serialised object
    data = req.dict()
    try:
      out = fn(**data)
      try:
        _ = json.dumps(out)
      except:
        response.status_code = 500
        return {"success": False, "message": "Function output cannot be serialised to JSON"}
      return {"success": True, "value": out}
    except Exception as e:
      response.status_code = 500
      return {"success": False, "message": str(e)}

  return generic_fwd_rest if _rest else generic_fwd


def nbx_py_rpc(op: Operator):
  base_model = create_model("nbx_py_rpc", rpc_name = (str, ""), key = (str, ""), value = (str, ""),)
  _nbx_py_rpc = NbxPyRpc(op)

  async def forward(req: base_model, response: Response):
    # no need to add serialisation because the NbPyRpc class will handle it
    data = req.dict()
    return _nbx_py_rpc(data, response)

  return forward


class NbxPyRpc(Operator):
  """This object is a shallow class that is used as a router of functions. Distributed computing combined with
  user friendliness of python means that some methods acn be routed and managed as long as there is a wire
  protocol. So here it is:

  ```python
  request = {
    "rpc_name": "__getattr__",
    "key": "string",            # always there
    "value": "b64-cloudpickle", # optional for functions
  }

  response = {
    "success": bool,
    "message": str,   # optional, will be error in case of failure
    "value": str,     # optional, will be b64-cloudpickle or might be empty ex. del
  }
  ```

  we should also add some routes to support a subset of important language features:

  - `__getattr__`: obtain any value by doing: `obj.x`
  - `__getitem__`: obtain any value by doing: `obj[x]`
  - `__setitem__`: set any value by doing: `obj[x] = y`
  - `__delitem__`: delete any value by doing: `del obj[x]`
  - `__iter__`: iterate over any iterable by doing: `for x in obj`
  - `__next__`: get next value from an iterator by doing: `next(obj)`
  - `__len__`: get length of any object by doing: `len(obj)`
  - `__contains__`: check if an object contains a value by doing: `x in obj`

  The reason we have chosen these for starting is that they can be used to represent any
  data structure required and get/set information from it. We can add more later like
  `__enter__` and `__exit__` to support context managers. Others like numerical operations
  `__add__` and `__sub__` doesn't really make sense. Maybe one day when we have neural networks
  but even then it's not clear how we would use them.
  """
  def __init__(self, op: Operator):
    super().__init__()
    self.wrapped_cls = op

  def forward(self, data, response: Response) -> Dict[str, str]:
    _k = set(tuple(data.keys())) - set(["rpc_name", "key", "value"])
    if _k:
      response.status_code = 400
      return {"success": False, "message": f"invalid keys: {_k}"}
    rpc_name = data.get("rpc_name", "")
    fn_map = {
      "__getattr__": (self.fn_getattr, key),
      "__getitem__": (self.fn_getitem, key),
      "__setitem__": (self.fn_setitem, key, value),
      "__delitem__": (self.fn_delitem, key),
      "__iter__": (self.fn_iter),
      "__next__": (self.fn_next),
      "__len__": (self.fn_len),
      "__contains__": (self.fn_contains, key),
    }
    _items = fn_map.get(rpc_name, None)
    if _items is None:
      response.status_code = 400
      return {"success": False, "message": f"invalid rpc_name: {rpc_name}"}

    key = data.get("key", "")
    value = data.get("value", "")

    if key:
      key = py_from_bs64(key)
    if value:
      value = py_from_bs64(value)

    fn, *args = _items
    try:
      out = fn(*args)
      return out
    except Exception as e:
      response.status_code = 500
      return {"success": False, "message": str(e)}

  def fn_getattr(self, key):
    if key.startswith("_"):
      return {"success": False, "message": f"cannot access private attributes starting with '_'"}
    out = getattr(self.wrapped_cls._op_wrap, key)
    return {"success": True, "value": py_to_bs64(out)}

  def fn_getitem(self, key):
    out = self.wrapped_cls._op_wrap[key]
    return {"success": True, "value": py_to_bs64(out)}

  def fn_setitem(self, key, value):
    self.wrapped_cls._op_wrap[key] = value
    return {"success": True}

  def fn_delitem(self, key):
    del self.wrapped_cls._op_wrap[key]
    return {"success": True}

  def fn_iter(self):
    out = iter(self.wrapped_cls._op_wrap)
    return {"success": True, "value": py_to_bs64(out)}

  def fn_next(self):
    out = next(self.wrapped_cls._op_wrap)
    return {"success": True, "value": py_to_bs64(out)}

  def fn_len(self):
    out = len(self.wrapped_cls._op_wrap)
    return {"success": True, "value": py_to_bs64(out)}

  def fn_contains(self, key):
    out = key in self.wrapped_cls._op_wrap
    return {"success": True, "value": py_to_bs64(out)}
