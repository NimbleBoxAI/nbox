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

  @abstractmethod
  def put_object(self, local_path: str, py_object) -> None:
    pass

  @abstractmethod
  def get_object(self, local_path: str) -> None:
    pass

  @abstractmethod
  def delete(self, local_path: str) -> None:
    pass

  @abstractmethod
  def list_files(self, local_path: str) -> None:
    pass

  @abstractmethod
  def upload_to(self, local_path: str, remote_path: str) -> None:
    pass

  @abstractmethod
  def download_from(self, local_path: str, remote_path: str) -> None:
    pass
