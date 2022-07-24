"""
These function are meant to make jobs easier and contain the code for what I am calling
``Subway``, which converts any OpenAPI spec into an RPC-like interface. At the end of the
day, each API call is really nothing but a functional call with added steps like
serialisation and networking combined. This kind of added complexity can:

#. Make code look really ugly, think try-catches and ``r.raise_for_request()``
#. Easy to develop because information has to be packaged as different parts for each\
    type of function call (GET, POST, ...). The true intent was always to pass it the\
    relevant information and forget about underlying details.
#. API endpoints are just strings and it's easy to have a typo in the URL. When the\
    ``Subway``is loaded with the OpenAPI spec, it will disallow incorrect URLs. while\
    managing the parameters correctly.
#. A more pythonic way of programming where you can use the dot notation.

Based os these ideas there are three types of subways:

#. ``Subway``: Does not load OpenAPI spec and will blindly call the API, avoid this
#. ``Sub30``: Built for OpenAPI v3.0.0, this becomes ``nbox_ws_v1``
#. ``SpecSubway``: This is used with the FastAPI's OpenAPI spec, this is used in systems\
    that use FastAPI (eg. Compute Server)
"""

import re
import time
import string
import threading
from requests import Session
from functools import lru_cache

from nbox.utils import logger

TIMEOUT_CALLS = 60


class Subway():
  """Simple code that allows extending things by ``.attr.ing`` them"""
  def __init__(self, _url, _session):
    self._url = _url.rstrip('/')
    self._session = _session

  def __repr__(self):
    return f"<Subway ({self._url})>"

  def __getattr__(self, attr):
    return Subway(f"{self._url}/{attr}", self._session)

  def __call__(self, method = "get", trailing = "", data = None, _verbose = False):
    fn = getattr(self._session, method)
    url = f"{self._url}{trailing}"
    if _verbose:
      logger.debug(f"Calling {url}")
    r = fn(url, json = data)
    if _verbose:
      logger.debug(r.content.decode())
    r.raise_for_status() # good when server is good
    return r.json()


@lru_cache
def filter_templates(paths):
  re_temps = []
  for x in paths:
    if "{" in x:
      temp_str = "^"
      for y in string.Formatter().parse(x):
        temp_str += y[0]
        if y[1]:
          temp_str += "\w+"
      re_temps.append((re.compile(temp_str + "$"), x,))
    else:
      re_temps.append((re.compile("^"+x+"$"), x,))
  return re_temps


class Sub30:
  def __init__(self, _url, _api, _session: Session, _default_key: str = "data", *, prefix = "", rl = None, bar = None):
    """Like Subway but built for Nimblebox Webserver APIs.

    Usage:

    .. code-block:: python

      ws = Sub30(
        _url = "https://my-web.site/",
        _session = nbox_session,
        _api = loads(fetch("https://my-web.site/openapi.json", True).decode()),
      )

    Args:
      _url (str): the base url of the API
      _api (dict): OpenAPI json dict
      _session (requests.Session): Session object to use for requests
      prefix (str, optional): This is internal, do not use it explicitly.
      last_call_at (float, optional): This is internal, do not use it explicitly.
    """
    self._url = _url.strip("/")
    self._api = _api
    self._session = _session
    self._default_key = _default_key
    self._prefix = prefix

    self._bar = bar or threading.Barrier(2)
    def _rate_limiter(s = 1.0):
      while True:
        self._bar.wait()
        time.sleep(s)

    if rl is None:
      # daemon meaning: it will die when the main thread dies
      # https://stackoverflow.com/questions/190010/daemon-threads-explanation
      rl = threading.Thread(target=_rate_limiter, daemon=True)
      rl.start()
    self._rl = rl # the limiter function

  def _create_keepalive(self):
    """Creates a keepalive thread. Need to define this as a function because sleeping
    can cause it to behave unlike a normal blocking program, so use this carefully."""

    def _keep_alive():
      # This is a hack to keep the session alive.
      # https://github.com/psf/requests/issues/4937
      # another more elegant solution is given here:
      # https://github.com/psf/requests/issues/4937#issuecomment-788899804
      while True:
        try:
          self._session.get(f"{self._url}")
        except Exception as e:
          pass
        
        if self.thread_kill_event.is_set():
          break
        time.sleep(30)

    self.thread = threading.Thread(target = _keep_alive)
    self.thread.start()
    self.thread_kill_event = threading.Event()

  def _stop(self):
    self.thread_kill_event.set()

  def __repr__(self):
    return f"<Sub30 ({self._url})>"

  def __getattr__(self, attr):
    return Sub30(f"{self._url}/{attr}", self._api, self._session, prefix=f"{self._prefix}/{attr}", rl = self._rl, bar = self._bar)

  def u(self, attr):
    return self.__getattr__(attr)

  def __call__(self, _method: str = None, **kwargs):
    r"""
    Args:
      _method (str, optional): if only one method is present this will be ignored else "get" will be used. Defaults to None.
    """
    paths = self._api["paths"]
    for v,s in enumerate(self._api["servers"]):
      # {url: "/api/v1"}, {url: "/api/v2"}
      setattr(self, f"v{v}", s["url"])
    params = None
    json = None

    ft = filter_templates(tuple(paths.keys()))
    path = None
    for t, index in ft:
      if re.match(t, self._prefix):
        path = index
        break
    if path == None:
      raise ValueError(f"No path found for '{self._prefix}'")

    # this is a match
    logger.debug("Matched path: " + path)
    p = paths[index]
    method = tuple(p.keys())[0] if len(p) == 1 else (
      _method if _method != None else "get"
    )
    body = p[method]

    if "parameters" in body and "requestBody" not in body:
      # likely a GET call
      params = kwargs

    if "requestBody" in body:
      # likely a POST call
      content_type = body["requestBody"]["content"]
      if "application/json" in content_type:
        json = {}
        schema_ref = content_type["application/json"]["schema"]
        for p in schema_ref["properties"]:
          if p in schema_ref.get("required", []):
            assert p in kwargs, f"{p} is required but not provided"
            json[p] = kwargs[p]
          else:
            if not p in kwargs and not "default" in schema_ref["properties"][p]:
              json[p] = None
            else:
              json[p] = kwargs[p]
      # elif "plain/text" in content_type:

    # call and return
    path = re.sub(r"\/_", "/", self._url)
    logger.debug(method.upper() + " " + path)
    
    # barrier will create a sync point for the rate limiter, i.e. if this is reached too quick
    # the rate limiter will sleep for the same amount of time, else it would already be waiting
    # for this
    self._bar.wait()
    r = self._session.request(method, path, json = json, params = params)
    try:
      r.raise_for_status()
    except Exception as e:
      logger.error(r.content.decode())
      raise e

    out = r.json()
    try:
      out = out[self._default_key]
    except KeyError:
      pass
    return out

  # _get = partial(__call__, _method = "get")
  # _post = partial(__call__, _method = "post")

class SpecSubway():
  def __init__(self, _url, _session, _spec, __name = None):
    """Subway but for fastAPI OpenAPI spec."""
    self._url = _url.rstrip('/')
    self._session = _session
    self._spec = _spec
    self._name = __name

    self._caller = (
      (len(_spec) == 3 and set(_spec) == set(["method", "meta", "src"])) or
      (len(_spec) == 4 and set(_spec) == set(["method", "meta", "src", "response_kwargs_dict"])) or
      "/" in self._spec
    )

  @classmethod
  def from_openapi(cls, openapi, _url, _session):
    logger.debug("Loading for OpenAPI version latest")
    paths = openapi["paths"]
    spec = openapi["components"]["schemas"]
    
    tree = {}
    for p in tuple(paths.keys()):
      t = tree
      for part in p.split('/')[1:]:
        part = "/" if part == "" else part
        t = t.setdefault(part, {})

    def _dfs(tree, trail = []):
      for t in tree:
        if tree[t] == {}:
          src = "/" + "/".join(trail)
          if t!= "/":
            src = src + "/" if src != "/" else src
            src = src + t
          
          try:
            data = paths[src]
          except:
            src = src + "/"
            data = paths[src]
          method = tuple(data.keys())[0]
          body = data[method]
          dict_ = {"method": method, "meta": None, "src": src}
          if "requestBody" in body:
            schema_ref = body["requestBody"]["content"]["application/json"]["schema"]["$ref"].split("/")[-1]
            _req_body = spec[schema_ref]
            kwargs_dict = list(_req_body["properties"])
            dict_["meta"] = {
              "kwargs_dict": kwargs_dict,
              "required": _req_body.get("required", None)
            }
          if "responses" in body:
            schema = body["responses"]["200"]["content"]["application/json"]["schema"]
            if "$ref" in schema:
              schema_ref = schema["$ref"].split("/")[-1]
              _req_body = spec[schema_ref]
              kwargs_dict = list(_req_body["properties"])
              if dict_["meta"] != None:
                dict_["meta"].update({"response_kwargs_dict": kwargs_dict})
              else:
                dict_["meta"] = {"response_kwargs_dict": kwargs_dict}
          tree[t] = dict_
        else:
          _dfs(tree[t], trail + [t])

    _dfs(tree)
    
    return cls(_url, _session, tree)

  def __repr__(self):
    return f"<SpecSubway ({self._url})>"

  def __getattr__(self, attr):
    # https://stackoverflow.com/questions/3278077/difference-between-getattr-vs-getattribute
    if self._caller and len(self._spec) == 1:
      raise AttributeError(f"'.{self._name}' does not have children")
    if attr not in self._spec:
      raise AttributeError(f"'.{attr}' is not a valid function")
    return SpecSubway(f"{self._url}/{attr}", self._session, self._spec[attr], attr)
  
  def __call__(self, *args, _verbose = False, _parse = False, **kwargs):
    from pprint import pprint
    pprint(self._spec)
    if not self._caller:
      raise AttributeError(f"'.{self._name}' is not an endpoint")
    spec = self._spec
    if self._caller and "/" in self._spec:
      spec = self._spec["/"]
    
    data = None
    if spec["meta"] == None:
      assert len(args) == len(kwargs) == 0, "This method does not accept any arguments"
    else:
      spec_meta = spec["meta"]
      if "kwargs_dict" not in spec_meta:
        assert len(args) == len(kwargs) == 0, "This method does not accept any arguments"
      else:
        kwargs_dict = spec["meta"]["kwargs_dict"]
        required = spec["meta"]["required"]
        data = {}
        for i in range(len(args)):
          if required != None:
            data[required[i]] = args[i]
          else:
            data[kwargs_dict[i]] = args[i]
        for key in kwargs:
          if key not in kwargs_dict:
            raise ValueError(f"{key} is not a valid argument")
          data[key] = kwargs[key]
        if required != None:
          for key in required:
            if key not in data:
              raise ValueError(f"{key} is a required argument")

    fn = getattr(self._session, spec["method"])
    url = f"{self._url}"
    if self._caller and "/" in self._spec:
      url += "/"
    if _verbose:
      logger.debug(f"{spec['method'].upper()} {url}")
      logger.debug("-->>", data)
    r = fn(url, json = data)
    if not r.status_code == 200:
      raise ValueError(r.content.decode())
    
    out = r.json()
    if _parse and self._spec["meta"] != None and "response_kwargs_dict" in self._spec["meta"]:
      out = [out[k] for k in self._spec["meta"]["response_kwargs_dict"]]
      if len(out) == 1:
        return out[0]
    return out

