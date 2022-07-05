import os
import zipfile
from datetime import datetime, timezone
from typing import Iterable

from nbox import Operator, logger
from nbox.instance import Instance


class FilesToInstance(Operator):
  def __init__(self, i: str, folder_name: str, workspace_id: str = None) -> None:
    """Transfer any number of files to a NimbleBox Instance at the `folder_name`.

    Args:
      i (str): Instance ID or Name
      folder_name (str): folder to transfer the zip object to
      workspace_id (str): Workspace ID or name, set to `None` for personal workspace
    """
    super().__init__()
    self.folder_name = folder_name
    logger.warn(f"Please ensure that the folder {self.folder_name} exists in '{i}'")
    self.instance = Instance(i, workspace_id)

  def forward(self, files: Iterable[str]):
    """Args:
        files (Iterable[str]): List of all the files to be transfered
    """
    # if not running, start the instance with the basic configuration, we just need to move the files
    self.instance.start()

    # pack all the items in a single zip file
    zip_name = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H-%M") + ".zip"
    logger.info(f"Packing to path: {zip_name}")
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zip_file:
      for f in files:
        zip_file.write(filename = f, arcname = f)

    # sanity check for the zip file
    os_stat = os.stat(zip_name)
    logger.info(f"Zip file size: {os_stat.st_size}")

    self.instance.mv(zip_name, f"nbx://{self.folder_name}/{zip_name}")
