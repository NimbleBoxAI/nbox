"""
This is the code for NBX-Relics which is a simple file system for your organisation.
"""
import os
import cloudpickle
import requests
import tabulate
from hashlib import md5
from typing import List
from copy import deepcopy
from functools import lru_cache

from nbox.auth import secret
from nbox.init import nbox_ws_v1
from nbox.utils import logger, env
from nbox.sublime.relics_rpc_client import (
  RelicStore_Stub,
  RelicFile,
  Relic as RelicProto,
  CreateRelicRequest,
  ListRelicFilesRequest,
  ListRelicsRequest
)
from nbox.relics.base import BaseStore

def get_relic_file(fpath: str, username: str, workspace_id: str):
  # assert os.path.exists(fpath), f"File {fpath} does not exist"
  # assert os.path.isfile(fpath), f"File {fpath} is not a file"

  # clean up fpath, remove any trailing slashes
  # trim any . or / from prefix and suffix
  fpath_cleaned = fpath.strip("./")

  extra = {}
  if os.path.exists(fpath):
    file_stat = os.stat(fpath)
    extra = {
      "created_on": int(file_stat.st_mtime),    # int
      "last_modified": int(file_stat.st_mtime), # int
      "size": max(1, file_stat.st_size),        # bytes
    }
  return RelicFile(
    name = fpath_cleaned,
    username = username,
    type = RelicFile.RelicType.FILE,
    workspace_id = workspace_id,
    **extra
  )


@lru_cache()
def _get_stub():
  url = "https://app.nimblebox.ai/relics"
  logger.debug("Connecting to RelicStore at: " + url)
  session = deepcopy(nbox_ws_v1._session)
  stub = RelicStore_Stub(url, session)
  return stub


def print_relics(workspace_id: str):
  stub = _get_stub()
  req = ListRelicsRequest(workspace_id = workspace_id,)
  out = stub.list_relics(req)
  headers = ["relic_name",]
  rows = [[r.name,] for r in out.relics]
  for l in tabulate.tabulate(rows, headers).splitlines():
    logger.info(l)


class RelicsNBX(BaseStore):
  list_relics = staticmethod(print_relics)

  def __init__(self, relic_name: str, workspace_id: str, create: bool = False):
    self.workspace_id = workspace_id
    self.relic_name = relic_name
    self.username = secret.get("username") # if its in the job then this part will automatically be filled
    self.stub = _get_stub()
    _relic = self.stub.get_relic_details(RelicProto(workspace_id=workspace_id, name=relic_name,))
    if  not _relic and create:
      # this means that a new one will have to be created
      logger.info(f"Creating new relic {relic_name}")
      self.relic = self.stub.create_relic(CreateRelicRequest(workspace_id=workspace_id, name = relic_name,))
      logger.info(f"Created new relic {self.relic}")
    else:
      self.relic = _relic

  def __repr__(self):
    return f"RelicStore({self.workspace_id}, {self.relic_name}, {'CONNECTED' if self.relic else 'NOT CONNECTED'})"

  def _upload_relic_file(self, local_path: str, relic_file: RelicFile):
    if not relic_file.relic_name:
      raise ValueError("relic_name not set in RelicFile")

    # ideally this is a lot like what happens in nbox
    logger.debug(f"Uploading {local_path} to {relic_file.name}")
    out = self.stub.create_file(_RelicFile = relic_file,)
    if not out.url:
      raise Exception("Could not get link")
    
    # do not perform merge here because "url" might get stored in MongoDB
    # relic_file.MergeFrom(out)
    logger.debug(f"URL: {out.url}")
    r = requests.post(
      url = out.url,
      data = out.body,
      files={
        "file": (out.body["key"], open(local_path, "rb"))
      }
    )
    logger.debug(f"Upload status: {r.status_code}")
    r.raise_for_status()

  def _download_relic_file(self, local_path: str, relic_file: RelicFile):
    if self.relic is None:
      raise ValueError("Relic does not exist, pass create=True")

    # ideally this is a lot like what happens in nbox
    logger.debug(f"Downloading {local_path} from S3 ...")
    out = self.stub.download_file(_RelicFile = relic_file,)
    if not out.url:
      raise Exception("Could not get link, are you sure this file exists?")
    
    # do not perform merge here because "url" might get stored in MongoDB
    # relic_file.MergeFrom(out)
    logger.debug(f"URL: {out.url}")
    with requests.get(out.url, stream=True) as r:
      r.raise_for_status()
      total_size = 0
      with open(local_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192): 
          # If you have chunk encoded response uncomment if
          # and set chunk_size parameter to None.
          #if chunk: 
          f.write(chunk)
          total_size += len(chunk)
    logger.debug(f"Download '{local_path}' status: OK ({total_size//1024} KB)")

  """
  At it's core the Relic is supposed to be a file system and not a client. Thus you cannot download something
  from a relic, but rather you tell the path you want to read and it will return the file. This is because of the
  fact that this is nothing but a glorified key value store.

  Moreover Relic as a broader concept is a storage in Von Neumann architecture. It is a storage that is not, so
  the more ways to store files built into it, the better the experience. So there are different types of put and
  get methods.
  """

  def put(self, local_path: str):
    """Put the file at this path into the relic"""
    if self.relic is None:
      raise ValueError("Relic does not exist, pass create=True")
    logger.info(f"Putting file: {local_path}")
    relic_file = get_relic_file(local_path, self.username, self.workspace_id)
    relic_file.relic_name = self.relic_name
    self._upload_relic_file(local_path, relic_file)

  def put_to(self, local_path: str, remote_path: str) -> None:
    if self.relic is None:
      raise ValueError("Relic does not exist, pass create=True")
    logger.info(f"Putting file: {local_path} to {remote_path}")
    relic_file = get_relic_file(local_path, self.username, self.workspace_id)
    relic_file.relic_name = self.relic_name
    relic_file.name = remote_path # override the name
    self._upload_relic_file(local_path, relic_file)

  def get(self, local_path: str):
    """Get the file at this path from the relic"""
    if self.relic is None:
      raise ValueError("Relic does not exist, pass create=True")
    logger.info(f"Getting file: {local_path}")
    relic_file = RelicFile(name = local_path.strip("./"),)
    relic_file.relic_name = self.relic_name
    relic_file.workspace_id = self.workspace_id
    self._download_relic_file(local_path, relic_file)

  def get_from(self, local_path: str, remote_path: str) -> None:
    if self.relic is None:
      raise ValueError("Relic does not exist, pass create=True")
    logger.info(f"Getting file: {local_path} from {remote_path}")
    relic_file = RelicFile(name = remote_path.strip("./"),)
    relic_file.relic_name = self.relic_name
    relic_file.workspace_id = self.workspace_id
    self._download_relic_file(local_path, relic_file)

  def rm(self, local_path: str):
    """Delete the file at this path from the relic"""
    if self.relic is None:
      raise ValueError("Relic does not exist, pass create=True")
    logger.info(f"Getting file: {local_path}")
    relic_file = get_relic_file(local_path, self.username, self.workspace_id)
    relic_file.relic_name = self.relic_name
    out = self.stub.delete_relic_file(relic_file)
    if not out.success:
      logger.error(out.message)
      raise ValueError("Could not delete file")

  def has(self, local_path: str):
    prefix, file_name = os.path.split(local_path)
    out = self.stub.list_relic_files(
      ListRelicFilesRequest(
        workspace_id=self.workspace_id,
        relic_name=self.relic_name,
        prefix=prefix,
        file_name=file_name
      )
    )
    for f in out.files:
      if f.name.strip("/") == local_path.strip("/"):
        return True
    return False

  """
  There are other convinience methods provided to keep consistency between the different types of relics. Note
  that we do no have a baseclass right now because I am note sure what are all the possible features we can have
  in common with all.
  """

  def put_object(self, key: str, py_object):
    """wrapper function for putting a python object"""
    # we will cache the object in the local file system
    cache_dir = os.path.join(env.NBOX_HOME_DIR(), ".cache")
    if not os.path.exists(cache_dir):
      os.makedirs(cache_dir)
    _key = os.path.join(cache_dir, md5(key.encode()).hexdigest())
    with open(_key, "wb") as f:
      cloudpickle.dump(py_object, f)
    self.put_to(_key, key)

  def get_object(self, key: str):
    """wrapper function for getting a python object"""
    cache_dir = os.path.join(env.NBOX_HOME_DIR(), ".cache")
    if not os.path.exists(cache_dir):
      os.makedirs(cache_dir)
    _key = os.path.join(cache_dir, md5(key.encode()).hexdigest())
    self.get_from(_key, key)
    with open(_key, "rb") as f:
      out = cloudpickle.load(f)
    return out

  """
  Some APIs are more on the level of the relic itself.
  """

  def delete(self):
    """Deletes your relic"""
    if self.relic is None:
      raise ValueError("Relic does not exist, nothing to delete")
    logger.warning(f"Deleting relic {self.relic_name}")
    self.stub.delete_relic(self.relic)

  def list_files(self, path: str = "") -> List[RelicFile]:
    """List all the files in the relic at path"""
    if self.relic is None:
      raise ValueError("Relic does not exist, pass create=True")
    logger.info(f"Listing files in relic {self.relic_name}")
    out = self.stub.list_relic_files(RelicFile(
      workspace_id = self.workspace_id,
      relic_name = self.relic_name,
      name = path
    ))
    return out.files

  def start_fs():
    # /my_relic/.....
    pass

# nbx jobs ... trigger --mount="dataset:/my-dataset/email/,model_master:/model"
