import os
import tabulate
from copy import deepcopy
from functools import lru_cache

from nbox.auth import secret, AuthConfig
from nbox.init import nbox_ws_v1
from nbox.utils import logger, get_mime_type
from nbox.relics.proto.relics_rpc_pb2 import ListRelicsRequest
from nbox.relics.proto.relics_pb2 import RelicFile
from nbox.relics.relics_rpc_client import RelicStore_Stub

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
def get_relics_stub():
  # url = "http://0.0.0.0:8081/relics" # debug mode
  url = secret("nbx_url") + "/relics"
  logger.debug("Connecting to RelicStore at: " + url)
  session = deepcopy(nbox_ws_v1._session)
  stub = RelicStore_Stub(url, session)
  return stub


def print_relics(workspace_id: str = ""):
  stub = get_relics_stub()
  workspace_id = workspace_id or secret(AuthConfig.workspace_id)
  req = ListRelicsRequest(workspace_id = workspace_id,)
  out = stub.list_relics(req)
  headers = ["relic_id", "relic_name",]
  rows = [[r.id, r.name,] for r in out.relics]
  for l in tabulate.tabulate(rows, headers).splitlines():
    logger.info(l)
