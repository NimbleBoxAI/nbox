import dill

from abc import abstractmethod

class BaseStore(object):
  @abstractmethod
  def put(self, local_path: str) -> None:
    pass

  @abstractmethod
  def get(self, local_path: str) -> None:
    pass

  @abstractmethod
  def rm(self, local_path: str) -> None:
    pass

  @abstractmethod
  def has(self, local_path: str) -> None:
    pass
