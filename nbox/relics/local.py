import os
import dill
from pathlib import Path
from typing import Any, Union

from nbox.utils import logger, FileLogger, env
from nbox.relics.base import BaseStore

class LocalStore(BaseStore):
  """
  Cache structure is like this:

  {cache_dir}/
    items/
      fdedc958b417adf63278938efa53c3b381f576446aede80bbc8f2c05320fcb4b
      1459d663d14b7b7ad82ebe5c98c8a0397b21d4e2b9f4711562746a5cb48f86c4
      ...
    _objects.bin # contains the entirity of information on this cache
    activity.log # contains the logs of this cache
  """
  def __init__(self, cache_dir: str = None, create: bool = True) -> Union[None, Any]:
    self.cache_dir = cache_dir or os.path.join(env.NBOX_HOME_DIR(), "relics")
    logger.info(f"Connecting object store: {self.cache_dir}")
    self._objects = {} # <key: item_path>
    self._objects_bin_path = f"{self.cache_dir}/_objects.bin"
    self._file_logger_path = f"{self.cache_dir}/activity.log"

    # format = "pickle/pyarrow/tfrecords" in _put means have to write in the _get

    # Create the cache directory if it doesn't exist
    if not os.path.exists(self.cache_dir):
      Path(self.cache_dir).mkdir(parents=create)
      # open(self._objects_bin_path, "w").close()
      # open(self._file_logger_path, "w").close()
    
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

  def _read_state(self):
    if os.path.exists(self._objects_bin_path):
      with open(self._objects_bin_path, "rb") as f:
        self._objects = dill.load(f)

  def _write_state(self):
    with open(self._objects_bin_path, "wb") as f:
      dill.dump(self._objects, f)
  
  def _put(self, key: str, value: bytes, ow: bool = False,) -> None:
    self._read_state()

    fp = f"{self.cache_dir}/{self._clean_key(key)}"
    _key = self.get_id(fp)
    item_path = f"{self.cache_dir}/items/{_key}"
    
    if os.path.exists(item_path) and not ow:
      raise Exception(f"File already exists: {fp}")
    with open(item_path, "wb") as f:
      dill.dump(value, f)

    # update the statae
    self._objects[fp] = item_path
    self._logs.info(f"PUT {key}")
    self._write_state()

  def _get(self, key: str) -> bytes:
    self._read_state()

    fp = f"{self.cache_dir}/{self._clean_key(key)}"
    item_path = self._objects.get(fp, None)
    if item_path is not None and os.path.exists(item_path):
      self._logs.info(f"GET {key}")
      with open(item_path, "rb") as f:
        out = dill.load(f)
      return out
    return None

  def _delete(self, key: str) -> None:
    self._read_state()

    fp = f"{self.cache_dir}/{self._clean_key(key)}"
    item_path = self._objects.get(fp, None)
    if item_path is not None and os.path.exists(item_path):
      os.remove(item_path)
      del self._objects[fp]
      self._logs.warning(f"DELETE {key}")
      self._write_state()
