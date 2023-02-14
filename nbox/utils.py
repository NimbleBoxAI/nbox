"""
Utility objects and functions.

This has a couple of cool things:

- `get_logger`: a master logger for nbox, this can be modified to log through anything
- `isthere`: a decorator that checks if a package is installed, if not it raises an error\
  It is more complicated than it needs to be because it is seedling for a way to package\
  functions and code together so that it can be used in a more dynamic way.
- `get_files_in_folder`: a function that returns all files in a folder with a certain extension
- `fetch`: a function that fetches a url and caches it in `tempdir` for faster loading
- `get_random_name`: a function that returns a random name, if `True` is passed returns\
  an `uuid4()` for truly random names :)
- `hash_`: a function that returns a hash of any python object, string is accurate, others\
  might be anything, but atleast it returns something.
- `folder/join`: to be used in pair, `join(folder(__file__), "server_temp.jinja")` means\
  relative path "../other.py". Scattered throughout the codebase, the super generic name will\
  bite me.
- `to_pickle/from_pickle`: to be used in pair, `to_pickle(obj, "path")` and `from_pickle("path")`\
  to save and load python objects to disk.
- `DBase`: सस्ता-protobuf (cheap protobuf), can be nested and `get_dict` will get for all\
  children
- `PoolBranch`: Or how to use multiprocessing, but blocked so you don't have to give a shit\
  about it.
"""

############################################################
# This file is d0 meaning that this has no dependencies!
# Do not import anything from rest of nbox here!
############################################################

# this file has bunch of functions that are used everywhere

import os
import sys
import json
import logging
import hashlib
import requests
import tempfile
import traceback
import randomname
import cloudpickle
from uuid import uuid4
from typing import List, Any
from contextlib import contextmanager
from base64 import b64encode, b64decode
from datetime import datetime, timezone
from pythonjsonlogger import jsonlogger
from functools import partial, lru_cache
from importlib.util import spec_from_file_location, module_from_spec
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from google.protobuf.timestamp_pb2 import Timestamp as Timestamp_pb

class env:
  """
  Single namespace for all environment variables.
  
  #. `NBOX_LOG_LEVEL`: Logging level for `nbox`, set `NBOX_LOG_LEVEL=info|debug|warning"
  #. `NBOX_ACCESS_TOKEN`: User token for NimbleBox.ai from [secrets](app.nimblebox.ai/secrets)
  #. `NBOX_NO_AUTH`: If set `secrets = None`, this will break any API request and is good only for local testing
  #. `NBOX_NO_LOAD_GRPC`: If set, will not load grpc stub
  #. `NBOX_NO_LOAD_WS`: If set, will not load webserver subway
  #. `NBOX_NO_CHECK_VERSION`: If set, will not check for version
  #. `NBOX_SSH_NO_HOST_CHECKING`: If set, `ssh` will not check for host key
  #. `NBOX_HOME_DIR`: By default `~/.nbx` folder, avoid changing this, user generally does not need to set this
  #. `NBOX_JSON_LOG`: Whether to print json-logs, user generally does not need to set this
  #. `NBOX_JOB_FOLDER`: Folder path for the job, user generally does not need to set this
  """
  # things user can chose to set if they want
  NBOX_LOG_LEVEL = lambda x: os.getenv("NBOX_LOG_LEVEL", x)
  NBOX_ACCESS_TOKEN = lambda x: os.getenv("NBOX_ACCESS_TOKEN", x)
  NBOX_SSH_NO_HOST_CHECKING = lambda x: os.getenv("NBOX_SSH_NO_HOST_CHECKING", x)

  # things that are good mostly for testing and development of nbox itself
  NBOX_NO_AUTH = lambda x: os.getenv("NBOX_NO_AUTH", x)
  NBOX_NO_LOAD_GRPC = lambda: os.getenv("NBOX_NO_LOAD_GRPC", False)
  NBOX_NO_LOAD_WS = lambda: os.getenv("NBOX_NO_LOAD_WS", False)
  NBOX_NO_CHECK_VERSION = lambda: os.getenv("NBOX_NO_CHECK_VERSION", False)
  
  # that that should be avoided to change, but provided here for max. control
  NBOX_HOME_DIR = lambda : os.environ.get("NBOX_HOME_DIR", os.path.join(os.path.expanduser("~"), ".nbx"))
  NBOX_JSON_LOG = lambda x: os.getenv("NBOX_JSON_LOG", x)
  NBOX_JOB_FOLDER = lambda x: os.getenv("NBOX_JOB_FOLDER", x)

  def set(key, value):
    os.environ[key] = value

  def get(key, default=None):
    return os.environ.get(key, default)

# logger /

logger = None

def get_logger():
  # add some handling so files can use functional way of getting logger
  global logger
  if logger is not None:
    return logger

  logger = logging.getLogger("utils")
  lvl = env.NBOX_LOG_LEVEL("info").upper()
  logger.setLevel(getattr(logging, lvl))

  if env.NBOX_JSON_LOG(False):
    logHandler = logging.StreamHandler()
    logHandler.setFormatter(jsonlogger.JsonFormatter(
      '%(timestamp)s %(levelname)s %(message)s ',
      timestamp=True
    ))
    logger.addHandler(logHandler)
  else:
    logHandler = logging.StreamHandler()
    logHandler.setFormatter(logging.Formatter(
      '[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
      datefmt = "%Y-%m-%dT%H:%M:%S%z"
    ))
    logger.addHandler(logHandler)

  return logger

logger = get_logger() # package wide logger

def log_traceback():
  logger.error(traceback.format_exc())

@contextmanager
def deprecation_warning(msg, remove, replace_by: str = None, help: str = None):
  from nbox.version import __version__
  msg = "Deprecation Warning" \
    f"\n  current: {__version__}" \
    f"\n  removed: {remove}" \
    f"\n      msg: {msg}"
  if replace_by:
    msg += f"\n  replace: {replace_by}"
  if help:
    msg += f"\n  help: {help}"
  logger.warning(msg)

class FileLogger:
  """Flush logs to a file, useful when we don't want to mess with current logging"""
  def __init__(self, filepath):
    self.filepath = filepath
    self.f = open(filepath, "a+")

    self.debug = partial(self.log, level="debug",)
    self.info = partial(self.log, level="info",)
    self.warning = partial(self.log, level="warning",)
    self.error = partial(self.log, level="error",)
    self.critical = partial(self.log, level="critical",)

  def log(self, message, level):
    self.f.write(f"[{datetime.now(timezone.utc).isoformat()}] {level}: {message}\n")
    self.f.flush()

# / logger

# lazy_loading/

def load_module_from_path(fn_name, file_path):
  spec = spec_from_file_location(fn_name, file_path)
  foo = module_from_spec(spec)
  mod_name = get_random_name()
  sys.modules[mod_name] = foo
  spec.loader.exec_module(foo)
  fn = getattr(foo, fn_name)
  return fn

def isthere(*packages, soft = True):
  """Checks all the packages

  Args:
      soft (bool, optional): If `False` raises `ImportError`. Defaults to `True`.
  """
  def wrapper(fn):
    _fn_ = fn
    def _fn(*args, **kwargs):
      # since we are lazy evaluating this thing, we are checking when the function
      # is actually called. This allows checks not to happen during __init__.
      for package in packages:
        # split the package name to get version number as well, not all will have package
        # version so continue those that have a version
        package = package.split("==")
        if len(package) == 1:
          package_name, package_version = package[0], None
        else:
          package_name, package_version = package
        if package_name in sys.modules:
          # trying to install these using pip will cause issues, so avoid that
          continue

        try:
          module = __import__(package_name)
          if hasattr(module, "__version__") and package_version:
            if module.__version__ != package_version:
              raise ImportError(f"{package_name} version mismatch")
        except ImportError:
          if not soft:
            raise ImportError(f"{package} is not installed, but is required by {fn}")
          # raise a warning, let the modulenotfound exception bubble up
          logger.warning(
            f"{package} is not installed, but is required by {fn.__module__}, some functionality may not work"
          )
      return _fn_(*args, **kwargs)
    return _fn
  return wrapper


# /lazy_loading

# path/

def get_files_in_folder(
  folder,
  ext = ["*"],
  abs_path: bool = True,
  followlinks: bool = False,
) -> List[str]:
  """Get files with `ext` in `folder`"""
  # this method is faster than glob
  import os
  all_paths = []
  _all = "*" in ext # wildcard means everything so speed up

  folder_abs = os.path.abspath(folder) if abs_path else folder
  for root,_,files in os.walk(folder_abs, followlinks=followlinks):
    if _all:
      all_paths.extend([join(root, f) for f in files])
      continue

    for f in files:
      for e in ext:
        if f.endswith(e):
          all_paths.append(os.path.join(root,f))
  return all_paths

def fetch(url, force = False):
  """Fetch and cache a url for faster loading, `force` re-downloads"""
  fp = join(tempfile.gettempdir(), hash_(url))
  if os.path.isfile(fp) and os.stat(fp).st_size > 0 and not force:
    with open(fp, "rb") as f:
      dat = f.read()
  else:
    dat = requests.get(url).content
    with open(fp + ".tmp", "wb") as f:
      f.write(dat)
    os.rename(fp + ".tmp", fp)
  return dat

def folder(x) -> str:
  """get the folder of this file path"""
  return os.path.split(os.path.abspath(x))[0]

def join(x, *args) -> str:
  """convienience function for os.path.join"""
  return os.path.join(x, *args)

def to_pickle(obj, path):
  """Save an object to a pickle file
  
  Args:
    obj: object to save
    path: path to save to
  """
  with open(path, "wb") as f:
    cloudpickle.dump(obj, f)

def from_pickle(path):
  """Load an object from a pickle file

  Args:
    path: path to load from
  """
  with open(path, "rb") as f:
    return cloudpickle.load(f)

def py_to_bs64(x: Any):
  return b64encode(cloudpickle.dumps(x)).decode("utf-8")

def py_from_bs64(x: bytes):
  return cloudpickle.loads(b64decode(x.encode("utf-8")))

def to_json(x: dict, fp: str = "", indent = 2):
  if fp:
    with open(fp, "w") as f:
      f.write(json.dumps(x, indent = indent))
  else:
    return json.dumps(x, indent = indent)

def from_json(x: str):
  if os.path.isfile(x):
    with open(x, "r") as f:
      return json.load(f)
  else:
    return json.loads(x)

def get_assets_folder():
  return join(folder(__file__), "assets")

# /path

# misc/

def get_random_name(uuid = False):
  """Get a random name, if `uuid` is `True`, return a uuid4"""
  if uuid:
    return str(uuid4())
  return randomname.generate()

def hash_(item, fn="md5"):
  """Hash sting of any item"""
  return getattr(hashlib, fn)(str(item).encode("utf-8")).hexdigest()


@lru_cache()
def get_mime_type(fp: str, defualt = "application/octet-stream"):
  with open(join(get_assets_folder(), "ex2mime.json"), "r") as f:
    mimes = json.load(f)
  return mimes.get(os.path.splitext(fp)[1].lower(), defualt)


# /misc

# datastore/

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

  def json(self, fp: str = "") -> str:
    if fp:
      with open(fp, "w") as f:
        f.write(json.dumps(self.get_dict()))
    else:    
      return json.dumps(self.get_dict())

# /datastore


################################################################################
# Parallel
# ========
# There already are many multiprocessing libraries for thread, core, pod, cluster
# but thee classes below are inspired by https://en.wikipedia.org/wiki/Collective_operation
#
# And that is why they are blocking classes, ie. it won't stop till all the tasks
# are completed. For reference please open the link above which has diagrams, there
# nodes can just be threads/cores/... Here is description for each of them:
#
# - Pool: apply same functions on different inputs
# - Branch: apply different functions on different inputs
################################################################################

# pool/

def threaded_map(fn, inputs, wait: bool = True, max_threads = 20, _name: str = None) -> None:
  """
  inputs is a list of tuples, each tuple is the input for single invocation of fn. order is preserved.
  """
  _name = _name or get_random_name(True)
  results = [None for _ in range(len(inputs))]
  with ThreadPoolExecutor(max_workers = max_threads, thread_name_prefix = _name) as exe:
    _fn = lambda i, x: [i, fn(*x)]
    futures = {exe.submit(_fn, i, x): i for i, x in enumerate(inputs)}
    if not wait:
      return futures
    for future in as_completed(futures):
      try:
        i, res = future.result()
        results[i] = res
      except Exception as e:
        raise e
  return results

# /pool

def hard_exit_program(code = 0):
  # why use os._exit over sys.exit:
  # https://stackoverflow.com/questions/9591350/what-is-difference-between-sys-exit0-and-os-exit0
  # https://stackoverflow.com/questions/19747371/python-exit-commands-why-so-many-and-when-should-each-be-used
  # tl;dr: os._exit kills without cleanup and so it's okay on the Pod
  os._exit(code)


class SimplerTimes:
  tz = timezone.utc

  def get_now_datetime() -> datetime:
    return datetime.now(SimplerTimes.tz)

  def get_now_float() -> float:
    return SimplerTimes.get_now_datetime().timestamp()

  def get_now_i64() -> int:
    return int(SimplerTimes.get_now_datetime().timestamp())

  def get_now_str() -> str:
    return SimplerTimes.get_now_datetime().strftime("%Y-%m-%d %H:%M:%S.%f")

  def get_now_pb():
    ts = Timestamp_pb()
    ts.GetCurrentTime()
    return ts
  
  def get_now_ns() -> int:
    return SimplerTimes.get_now_pb().ToNanoseconds()

  def i64_to_datetime(i64) -> datetime:
    return datetime.fromtimestamp(i64, SimplerTimes.tz)
