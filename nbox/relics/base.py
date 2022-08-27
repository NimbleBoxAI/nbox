from abc import abstractmethod
from typing import List

class BaseStore(object):
  """This is the base class for all Relic Stores. Instead of treating these functions as just means to an
  objective, we treat them as a ideas to organise things."""
  @abstractmethod
  def put(self, local_path: str) -> None:
    """Puts whatever you have at local_path to the same path in the store"""
    pass

  @abstractmethod
  def put_object(self, local_path: str, py_object) -> None:
    """Convinience function for putting a python object in the store"""
    pass

  @abstractmethod
  def put_to(self, local_path: str, remote_path: str) -> None:
    """like put but with a different remote path"""
    pass

  @abstractmethod
  def list_files(self, local_path: str) -> List[str]:
    """list all files in this relic"""
    pass

  @abstractmethod
  def has(self, local_path: str) -> bool:
    """check if this file exists in this relic"""
    pass

  @abstractmethod
  def get(self, local_path: str) -> None:
    """gets the put-ed file from the store to local_path"""
    pass

  @abstractmethod
  def get_object(self, local_path: str) -> None:
    """Convinience function for getting a python object from the store"""
    pass

  @abstractmethod
  def get_from(self, local_path: str, remote_path: str) -> None:
    """like get but with a different remote path"""
    pass

  @abstractmethod
  def rm(self, local_path: str) -> None:
    """delete this file in this relic, should not have side efects, meaning relic should only delete the file
    in the areas of its control. ex: If a file is ``put`` on relics, deleting it should not delete the file
    locally."""
    pass

  @abstractmethod
  def delete(self) -> None:
    """delete the relic"""
    pass
