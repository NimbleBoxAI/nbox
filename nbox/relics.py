"""
This is the code for NBX-Relics which is a simple file system for your organisation.
"""
import os
import time
import json
import cloudpickle
import requests
import tabulate
from hashlib import md5
from typing import List
from copy import deepcopy
from subprocess import Popen
from functools import lru_cache

from nbox.auth import secret, AuthConfig
from nbox.init import nbox_ws_v1
from nbox.messages import message_to_dict
from nbox.utils import logger, env, get_mime_type
from nbox.sublime.relics_rpc_client import (
  RelicStore_Stub,
  RelicFile,
  Relic as RelicProto,
  CreateRelicRequest,
  ListRelicFilesRequest,
  ListRelicsRequest,
  BucketMetadata
)

def get_relic_file(fpath: str, username: str, workspace_id: str = ""):
  workspace_id = workspace_id or secret(AuthConfig.workspace_id)
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
  content_type = get_mime_type(fpath_cleaned, "application/octet-stream")
  return RelicFile(
    name = fpath_cleaned,
    username = username,
    type = RelicFile.RelicType.FILE,
    workspace_id = workspace_id,
    content_type=content_type,
    **extra
  )


@lru_cache()
def _get_stub():
  # url = "http://0.0.0.0:8081/relics" # debug mode
  url = secret("nbx_url") + "/relics"
  logger.debug("Connecting to RelicStore at: " + url)
  session = deepcopy(nbox_ws_v1._session)
  stub = RelicStore_Stub(url, session)
  return stub

def print_relics(workspace_id: str = ""):
  stub = _get_stub()
  workspace_id = workspace_id or secret(AuthConfig.workspace_id)
  req = ListRelicsRequest(workspace_id = workspace_id,)
  out = stub.list_relics(req)
  headers = ["relic_id", "relic_name",]
  rows = [[r.id, r.name,] for r in out.relics]
  for l in tabulate.tabulate(rows, headers).splitlines():
    logger.info(l)


class UserAgentType:
  PYTHON_REQUESTS = "python-requests"
  CURL = "curl"

  def all():
    return [UserAgentType.PYTHON_REQUESTS, UserAgentType.CURL,]


class Relics():
  list = staticmethod(print_relics)

  def __init__(
    self,
    relic_name: str,
    workspace_id: str = "",
    create: bool = False,
    prefix: str = "",
    *,
    bucket_name: str = "",
    region: str = "",
    nbx_resource_id: str = "",
    nbx_integration_token: str = "",
  ):
    """
    The client for NBX-Relics. Auto switches to different user/agents for upload download

    Args:
      relic_name (str): The name of the relic.
      workspace_id (str): The workspace ID, if not provided, will be one in global config.
      create (bool): Create the relic if it does not exist.
      prefix (str): The prefix to use for all files in this relic. If provided all the files are uploaded and downloaded with this prefix.
    """
    self.workspace_id = workspace_id or secret(AuthConfig.workspace_id)
    self.relic_name = relic_name
    self.username = secret("username") # if its in the job then this part will automatically be filled
    self.prefix = prefix.strip("/")
    self.stub = _get_stub()
    for _ in range(2):
      _relic = self.stub.get_relic_details(RelicProto(workspace_id=self.workspace_id, name=relic_name,))
      if _relic != None:
        break
      time.sleep(1)

    # print("asdfasdfasdfasdf", _relic, not _relic and create)
    if not _relic and create:
      # this means that a new one will have to be created
      logger.debug(f"Creating new relic {relic_name}")
      self.relic = self.stub.create_relic(CreateRelicRequest(
        workspace_id=self.workspace_id,
        name = relic_name,
        bucket_meta = BucketMetadata(
          bucket_name = bucket_name,
          region = region,
          backend = BucketMetadata.Backend.AWS_S3,
        ),
        nbx_resource_id = nbx_resource_id,
        nbx_integration_token = nbx_integration_token,
      ))
      logger.debug(f"Created new relic {self.relic}")
    else:
      self.relic = _relic
    
    self.uat = UserAgentType.PYTHON_REQUESTS

  def set_user_agent(self, user_agent_type: str):
    if user_agent_type not in UserAgentType.all():
      raise ValueError(f"Invalid user agent type: {user_agent_type}")
    logger.info(f"Setting user agent to {user_agent_type} from {self.uat}")
    self.uat = user_agent_type

  def __repr__(self):
    return f"Relics({self.relic_name}, {'CONNECTED' if self.relic else 'NOT CONNECTED'}" + \
      (f", prefix={self.prefix}" if self.prefix else "") + \
      ")"

  def _upload_relic_file(self, local_path: str, relic_file: RelicFile):
    if not relic_file.relic_name:
      raise ValueError("relic_name not set in RelicFile")
    if self.prefix:
      relic_file.name = f"{self.prefix}/{relic_file.name}"

    # ideally this is a lot like what happens in nbox
    logger.info(f"Uploading {local_path} to {relic_file.name}")
    for _ in range(2):
      out = self.stub.create_file(_RelicFile = relic_file,)
      if out != None:
        break
      time.sleep(1)

    if not out.url:
      raise Exception("Could not get link")

    # do merge 'out' and 'relic_file' here because "url" might get stored in MongoDB
    # relic_file.MergeFrom(out)
    ten_mb = 10 ** 7
    uat = self.uat
    old_uat = uat
    if out.size > ten_mb:
      logger.warning(f"File {local_path} is larger than 10 MiB ({out.size} bytes), this might take a while")
      logger.warning(f"Switching to user/agent: cURL for this upload")
      uat = UserAgentType.CURL

    if uat == UserAgentType.PYTHON_REQUESTS:
      logger.debug(f"URL: {out.url}")
      logger.debug(f"body: {out.body}")
      r = requests.post(
        url = out.url,
        data = out.body,
        files = {"file": (out.body["key"], open(local_path, "rb"))}
      )
      logger.info(f"Upload status: {r.status_code}")
      r.raise_for_status()
    elif uat == UserAgentType.CURL:
      # TIL: https://stackoverflow.com/a/58237351
      # the fields in the post can be sent in any order

      import shlex
      shell_com = f'curl -X POST -F key={out.body["key"]} '
      for k,v in out.body.items():
        if k == "key":
          continue
        shell_com += f'-F {k}={v} '
      shell_com += f'-F file="@{local_path}" {out.url}'
      logger.debug(f"Running shell command: {shell_com}")
      Popen(shlex.split(shell_com)).wait()

    if old_uat != uat:
      logger.warning(f"Restoring user/agent to {old_uat}")
      self.set_user_agent(old_uat)


  def _download_relic_file(self, local_path: str, relic_file: RelicFile):
    if self.relic is None:
      raise ValueError("Relic does not exist, pass create=True")
    if self.prefix:
      relic_file.name = self.prefix + "/" + relic_file.name

    # ideally this is a lot like what happens in nbox
    logger.debug(f"Downloading {local_path} from S3 ...")
    for _ in range(2):
      out = self.stub.download_file(_RelicFile = relic_file,)
      if out != None:
        break
      time.sleep(1)

    if not out.url:
      raise Exception("Could not get link, are you sure this file exists?")

    # same logic as in upload but for download
    ten_mb = 10 ** 7
    uat = self.uat
    if out.size > ten_mb:
      logger.warning(f"File {local_path} is larger than 10 MiB ({out.size} bytes), this might take a while")
      logger.warning(f"Switching to user/agent: cURL for this download")
      uat = UserAgentType.CURL

    if uat == UserAgentType.PYTHON_REQUESTS:    
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
    elif uat == UserAgentType.CURL:
      import shlex
      shell_com = f'curl -o {local_path} {out.url}'
      logger.debug(f"Running shell command: {shell_com}")
      Popen(shlex.split(shell_com)).wait()
      total_size = os.path.getsize(local_path)
    logger.debug(f"Download '{local_path}' status: OK ({total_size//1000} KiB)")

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
    logger.debug(f"Putting file: {local_path}")
    relic_file = get_relic_file(local_path, self.username, self.workspace_id)
    relic_file.relic_name = self.relic_name
    self._upload_relic_file(local_path, relic_file)

  def put_to(self, local_path: str, remote_path: str) -> None:
    if self.relic is None:
      raise ValueError("Relic does not exist, pass create=True")
    logger.debug(f"Putting file: {local_path} to {remote_path}")
    relic_file = get_relic_file(local_path, self.username, self.workspace_id)
    relic_file.relic_name = self.relic_name
    relic_file.name = remote_path # override the name
    self._upload_relic_file(local_path, relic_file)

  def get(self, local_path: str):
    """Get the file at this path from the relic"""
    if self.relic is None:
      raise ValueError("Relic does not exist, pass create=True")
    logger.debug(f"Getting file: {local_path}")
    relic_file = RelicFile(name = local_path.strip("./"),)
    relic_file.relic_name = self.relic_name
    relic_file.workspace_id = self.workspace_id
    self._download_relic_file(local_path, relic_file)

  def get_from(self, local_path: str, remote_path: str, unzip: bool = False) -> None:
    # TODO:@yashbonde add support for unzipping
    if self.relic is None:
      raise ValueError("Relic does not exist, pass create=True")
    logger.debug(f"Getting file: {local_path} from {remote_path}")
    relic_file = RelicFile(name = remote_path.strip("./"),)
    relic_file.relic_name = self.relic_name
    relic_file.workspace_id = self.workspace_id
    self._download_relic_file(local_path, relic_file)

  def rm(self, remote_path: str):
    """Delete the file at this path from the relic"""
    if self.relic is None:
      raise ValueError("Relic does not exist, pass create=True")
    logger.debug(f"Getting file: {remote_path}")
    relic_file = get_relic_file(remote_path, self.username, self.workspace_id)
    relic_file.relic_name = self.relic_name
    for _ in range(2):
      out = self.stub.delete_relic_file(relic_file)
      if out != None:
        break
      time.sleep(1)
    
    if not out.success:
      logger.error(out.message)
      raise ValueError("Could not delete file")

  def has(self, path: str):
    prefix, file_name = os.path.split(path)
    for _ in range(2):
      out = self.stub.list_relic_files(
        ListRelicFilesRequest(
          workspace_id=self.workspace_id,
          relic_name=self.relic_name,
          prefix=prefix,
          file_name=file_name
        )
      )
      if out != None:
        break
      time.sleep(1)

    for f in out.files:
      if f.name.strip("/") == path.strip("/"):
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

  def list_files(self, path: str = "") -> RelicFile:
    """List all the files in the relic at path"""
    if self.relic is None:
      raise ValueError("Relic does not exist, pass create=True")
    logger.debug(f"Listing files in relic {self.relic_name}")
    for _ in range(2):
      p = self.prefix
      if path:
        p += "/" + path
      out = self.stub.list_relic_files(ListRelicFilesRequest(
        workspace_id = self.workspace_id,
        relic_id = self.relic.id,
        prefix = p
      ))
      if out != None:
        break
      time.sleep(1)
    
    return out


# nbx jobs ... trigger --mount="dataset:/my-dataset/email/,model_master:/model"
