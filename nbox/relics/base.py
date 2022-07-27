import re
from uuid import uuid4
from hashlib import sha256
from typing import Any, Tuple

class BaseStore(object):
  # Typical OOPs things

  def _get_uuid(self, s = True) -> str:
    if s:
      return str(uuid4()).split('-')[0]
    return str(uuid4())

  def get_id(self, key: str, random: bool = False) -> str:
    _key = sha256(key.encode('utf-8')).hexdigest()
    if random:
      return str(uuid4()).split('-')[0] + "-" + _key
    return _key

  def _clean_key(self, key: str) -> str:
    return key.strip('/')

  def _put(self, key: str, value: Any) -> None:
    raise NotImplementedError("Each store to be used must implement _put")

  def put(self, key: str, value: Any, ow: bool = False) -> None:
    self._put(key, value, ow)

  def _get(self, key: str) -> Any:
    raise NotImplementedError("Each store to be used must implement _get")

  def get(self, key: str, default = None) -> Any:
    out = self._get(key)
    if out is None:
      return default
    return out

  def __getitem__(self, key: str):
    return self.get(key)

  def _delete(self, key: str) -> None:
    raise NotImplementedError("Each store to be used must implement _delete")

  def delete(self, key: str) -> None:
    self._delete(key)

