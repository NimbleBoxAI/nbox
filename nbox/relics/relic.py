import re
from enum import Enum

from nbox.init import nbox_ws_v1
from nbox.relics.base import LocalStore

class RelicTypes(Enum):
  """
  Types of relics.
  """
  UNSET = 0

  # all the files in this mode are going to be stored in some directory
  LOCAL = 1

  # this mode means that relic is the one run by NimbleBox
  NBX = 0

  # the backend in this case is the AWS S3
  AWS_S3 = 3


class Relics():
  _mode = RelicTypes.UNSET
  def __init__(self, id_or_url, workspace_id = "personal", **kwargs):
    # go over the bunch of cases that id_or_url can be and set the mode
    if id_or_url.startswith('http'):
      out = re.findall(r".ai\/(\w+)\/relics\/(.*)$", id_or_url)
      if not out:
        raise Exception(f"Invalid URL: '{id_or_url}'")
      self.workspace_id = out[0][0]
      self.relic_id = out[0][1]
      self.url = id_or_url

    elif id_or_url.startswith('local:'):
      self.store = LocalStore(id_or_url[6:], **kwargs)
      self._mode = RelicTypes.LOCAL # change the mode to local

    elif re.findall(r"^\w{8}$", id_or_url):
      self.workspace_id = workspace_id
      self.relic_id = id_or_url

    # create the subway stub
    # self.stub = nbox_ws_v1.u(workspace_id).relics.u(self.relic_id)

    # hit NBX APIs to get more information about this
    # self.refresh()

  def refresh(self):
    data = self.stub()
    
    builder = data["builder"] # some information about the builder like cloud vendor, etc
    api_meta = data["vendor_metdata"] # say things that I get in get_upload_url()
    file_meta = data["file_meta"] # folder also are just file references

    self.name = file_meta["name"]
    self.is_file = file_meta["is_file"]
    self.size = file_meta["size"]

  def download(self, key, local_path):
    """download and store the file to local_path"""
    pass

  def upload(self, local_path, key = None):
    """upload the file to NBX"""
    pass

  def delete(self, key):
    """delete the file from NBX"""
    pass
