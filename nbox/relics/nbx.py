"""
This is the code for NBX-Relics which is a simple management tool for 
"""
from copy import deepcopy

from nbox.auth import secret
from nbox.init import nbox_ws_v1
from nbox.utils import logger
from nbox.sublime.relics_rpc_client import *

def get_relic_file(fpath: str, username: str, workspace_id: str):
  assert os.path.exists(fpath), f"File {fpath} does not exist"
  assert os.path.isfile(fpath), f"File {fpath} is not a file"

  # clean up fpath, remove any trailing slashes
  # trim any . or / from prefix and suffix
  fpath_cleaned = fpath.strip("./")
  file_stat = os.stat(fpath)
  return RelicFile(
    name = fpath_cleaned,
    created_on = int(file_stat.st_mtime),    # int
    last_modified = int(file_stat.st_mtime), # int
    size = max(1, file_stat.st_size),        # bytes
    username = username,
    type = RelicFile.RelicType.FILE,
    workspace_id = workspace_id,
  )

class NBXRelicStore():
  def __init__(self, workspace_id: str, relic_name: str, create: bool = False):
    self.workspace_id = workspace_id
    self.relic_name = relic_name
    self.username = secret.get("username") # if its in the job then this part will automatically be filled

    # relic_path = nbox_ws_v1.u(workspace_id).relics # v1/{workspace_id}/relics/
    # self.stub = RelicStore_Stub(relic_path._url, deepcopy(relic_path._session))    
    self.stub = RelicStore_Stub("http://0.0.0.0:8081", deepcopy(nbox_ws_v1._session))    
    
    _relic = self.stub.get_relic_details(
      Relic(workspace_id=workspace_id, name=relic_name,)
    )
    if _relic is None and create:
      # this means that a new one will have to be created
      logger.info(f"Creating new relic {relic_name}")
      self.relic = self.stub.create_relic(
        CreateRelicRequest(workspace_id=workspace_id, name = relic_name,)
      )
    else:
      self.relic = _relic

  def __repr__(self):
    return f"RelicStore({self.workspace_id}, {self.relic})"

  def upload_relic_file(self, local_path: str, relic_file: RelicFile):
    if not relic_file.relic_name:
      raise ValueError("relic_name not set in RelicFile")

    # ideally this is a lot like what happens in nbox
    logger.debug(f"Uploading {local_path} to {relic_file.name}")
    out = self.stub.create_file(_RelicFile = relic_file,)
    
    # do not perform merge here because "url" might get stored in MongoDB
    # relic_file.MergeFrom(out)
    logger.debug(f"URL: {out.url}")
    with open(local_path, "rb") as f:
      r = requests.put(out.url, data = f)
    logger.debug(f"Upload status: {r.status_code}")

  def download_relic_file(self, local_path: str, relic_file: RelicFile):
    if self.relic is None:
      raise ValueError("Relic does not exist! pass create=True")

    # ideally this is a lot like what happens in nbox
    logger.debug(f"Downloading {local_path} from S3 ...")
    out = self.stub.download_file(_RelicFile = relic_file,)
    
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
    logger.info("Download status: OK")
    logger.info(f"    filepath: {local_path}")
    logger.info(f"  total_size: {total_size//1024} KB")

  def _put(self, local_path: str):
    if self.relic is None:
      raise ValueError("Relic does not exist! pass create=True")

    # get the file
    relic_file = get_relic_file(local_path, self.username, self.workspace_id)
    relic_file.relic_name = self.relic_name
    self.upload_relic_file(local_path, relic_file)
  
  def _get(self, local_path: str):
    if self.relic is None:
      raise ValueError("Relic does not exist! pass create=True")

    # get the file
    relic_file = get_relic_file(local_path, self.username, self.workspace_id)
    relic_file.relic_name = self.relic_name
    self.download_relic_file(local_path, relic_file)
