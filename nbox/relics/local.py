import os
import dill
from glob import glob
from uuid import uuid4
from pathlib import Path
from shutil import rmtree
from hashlib import sha256
from typing import Any, Union, List

from nbox.utils import logger, FileLogger, env
from nbox.relics.base import BaseStore

class RelicLocal(BaseStore):
  """
  The Relic is a part of a filesystem, however `RelicLocal` is an exception since the data that it recieves might
  optionally be a python object in which case it needs to store that.

  Cache structure is like this:

  {cache_dir}/
    _objects.bin # contains the entirity of information on this cache
    activity.log # contains the logs of this cache
    items/
      fdedc958b417adf63278938efa53c3b381f576446aede80bbc8f2c05320fcb4b
      1459d663d14b7b7ad82ebe5c98c8a0397b21d4e2b9f4711562746a5cb48f86c4
      ...
  """
  def __init__(self, relic_name: str, workspace_id: str, create: bool = False):
    self.relic_name = relic_name
    self.workspace_id = workspace_id
    self.cache_dir = os.path.join(env.NBOX_HOME_DIR(), "relics")
    logger.info(f"Connecting object store: {self.cache_dir}")
    self._objects = {} # <key: item_path>
    self._objects_bin_path = f"{self.cache_dir}/_objects.bin"
    self._file_logger_path = f"{self.cache_dir}/activity.log"

    # Create the cache directory if it doesn't exist
    if not os.path.exists(self.cache_dir):
      Path(self.cache_dir).mkdir(parents=create)
    
    # load the data from the cache
    if os.path.isdir(self.cache_dir):
      if not os.path.exists(self._objects_bin_path):
        self._objects = {}
      else:
        with open(self._objects_bin_path, "rb") as f:
          self._objects = dill.load(f)
      if not os.path.exists(f"{self.cache_dir}/items"):
        os.makedirs(f"{self.cache_dir}/items")
      self._logs = FileLogger(self._file_logger_path)
    else:
      raise Exception(f"{self.cache_dir} is not a directory")

  def get_id(self, key: str = "") -> str:
    _key = sha256(key.encode('utf-8')).hexdigest()
    item_path = f"{self.cache_dir}/items/{_key}" # the path to the actual file
    object_key = f"{self.cache_dir}/relics/{self.relic_name}/{_key}"
    return (object_key, item_path)

  def _read_state(self):
    if os.path.exists(self._objects_bin_path):
      with open(self._objects_bin_path, "rb") as f:
        self._objects = dill.load(f)

  def _write_state(self):
    with open(self._objects_bin_path, "wb") as f:
      dill.dump(self._objects, f)

  """
  Standard set of APIs for put, get, rm, has.
  """

  def put(self, key):
    """The put in case of a local relic only updates the internal map to the key filepath"""
    self._read_state()
    object_key, _ = self.get_id(key)
    self._objects[object_key] = key # reference to existing object
    self._logs.info(f"[{self.workspace_id}/{self.relic_name}] PUT {key}")
    self._write_state()

  def get(self, key: str) -> None:
    self._read_state()
    object_key, _ = self.get_id(key) # internal path
    item_path = self._objects.get(object_key, None)
    if item_path is not None and os.path.exists(item_path):
      self._logs.info(f"[{self.workspace_id}/{self.relic_name}] GET {key}")
    elif (item_path is not None) and (not os.path.exists(item_path)):
      raise Exception("Trying to get missing file")
    else:
      raise Exception("Could not get link, are you sure this file exists?")

  def rm(self, key: str) -> None:
    self._read_state()
    object_key, _ = self.get_id(key) # internal path
    item_path = self._objects.get(object_key, None)
    if item_path is not None and os.path.exists(item_path):
      os.remove(item_path)
      del self._objects[object_key]
      self._logs.warning(f"[{self.workspace_id}/{self.relic_name}] DEL {key}")
      self._write_state()
    elif (item_path is not None) and (not os.path.exists(item_path)):
      raise Exception("Trying to get missing file")
    else:
      raise Exception("Could not get link, are you sure this file exists?")

  def has(self, key: str) -> None:
    self._read_state()
    object_key, _ = self.get_id(key) # internal path
    item_path = self._objects.get(object_key, None)
    if item_path is not None and os.path.exists(item_path):
      return True
    return False

  """
  Conviniece for python objects.
  """

  def put_object(self, key: str, value: bytes) -> None:
    """The put in case of a local relic only updates the internal map to the key filepath"""
    self._read_state()
    object_key, item_path = self.get_id(key) # internal path
    with open(item_path, "wb") as f:
      dill.dump(value, f)
    self._objects[object_key] = item_path # the path is the actual place to store
    self._logs.info(f"[{self.workspace_id}/{self.relic_name}] PUTO {key}")
    self._write_state()

  def get_object(self, key: str) -> bytes:
    self._read_state()
    object_key, _ = self.get_id(key) # internal path
    item_path = self._objects.get(object_key, None)
    if item_path is not None and os.path.exists(item_path):
      self._logs.info(f"[{self.workspace_id}/{self.relic_name}] GETO {key}")
      with open(item_path, "rb") as f:
        out = dill.load(f)
      return out
    elif (item_path is not None) and (not os.path.exists(item_path)):
      raise Exception("Trying to get missing file")
    else:
      raise Exception("Could not get link, are you sure this file exists?")

  """
  Some APIs are more on the level of the relic itself.
  """

  def delete(self):
    """Deletes your relic, which in this case means it removes all the entries at that prefix"""
    self._read_state()
    self._logs.warning(f"[{self.workspace_id}/{self.relic_name}] DELR {self.relic_name}")
    object_key, _ = self.get_id() # internal path to relic
    object_relic_key = "/".join(object_key.split("/")[:-1])
    done = False
    for key, item_path in self._objects.items():
      if key.startswith(object_relic_key):
        self.rm(item_path)
        done = True
    if not done:
      raise Exception("Could not delete relic, are you sure it exists?")
    self._write_state()

  def list_files(self, path: str = "") -> List[str]:
    """List all the files in the relic at path"""
    path = path.strip("/")
    object_key, _ = self.get_id(path) # internal path to relic
    object_relic_key = "/".join(object_key.split("/")[:-1])

    # keep only those files that either have the same name or only till one depth ('/')
    all_f = [k for k in self.objects.keys() if k.startswith(object_relic_key)]
    files = set()
    total_slashes = len(object_relic_key.split("/"))
    for f in all_f:
      sections = f.split("/")
      sections = sections[:total_slashes+1]
      files.add("/".join(sections))
    return list(files)
