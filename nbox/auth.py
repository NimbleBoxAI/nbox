"""
This code is used to manage the authentication of the entire ``nbox`` package. For authentication
it will create a ``.nbx`` in the user's home directory (``~/.nbx``, in case of linux) and store
a file called ``secrets.json``. This folder will also contain more information and files that
are used elsewhere as well ex. files generated when takling to any instance.

Users can chose to set configs at a global level for convenience. Here's a list of configs that can be set:

global.workspace_id = '' # this will set the default workspace id for all commands, users can override this by
  passing the ``--workspace-id`` flag
"""
import os
import json
import requests
import webbrowser
from getpass import getpass
from enum import Enum

import nbox.utils as U
from nbox.utils import join, logger


class ConfigString(Enum):
  workspace_id = "config.global.workspace_id"

class NBXClient:
  def __init__(self, nbx_url = "https://app.nimblebox.ai"):
    """
    Single source for all the secrets file. Can persist across multiple processes.
    """
    os.makedirs(U.env.NBOX_HOME_DIR(), exist_ok=True)
    self.fp = join(U.env.NBOX_HOME_DIR(), "secrets.json")

    access_token = U.env.NBOX_USER_TOKEN("")

    # if this is the first time starting this then get things from the nbx-hq
    if not os.path.exists(self.fp):
      if not access_token:
        logger.info(f"Ensure that you put the email ID you have signed up with!")
        _secrets_url = f"{nbx_url}/secrets"
        logger.info(f"Opening: {_secrets_url}")
        webbrowser.open(_secrets_url)
        access_token = getpass("Access Token: ")
      
      # Once we have the access token, we can get the secrets
      r = requests.get(f"{nbx_url}/api/v1/user/account_details", headers={"Authorization": f"Bearer {access_token}"})
      r.raise_for_status()
      try:
        username = r.json()["data"]["username"]
        email = r.json()["data"]["email"]
      except Exception as e:
        logger.error(f"Could not get the username and email from the response")
        logger.error(f"This should not have happened, please contact NimbleBox support.")
        raise e

      workspace_id = input("Set the default workspace ID (blank means personal): ").strip()
      if not workspace_id:
        workspace_id = "personal"
      logger.info(f"Setting the default workspace ID to: {workspace_id}")

      # create the objects
      self.secrets = {
        "email": email,
        "access_token": access_token,
        "nbx_url": nbx_url,
        "username": username,

        # config values that can be set by the user for convenience
        ConfigString.workspace_id.value: workspace_id
      }
      with open(self.fp, "w") as f:
        f.write(repr(self))
      logger.info("Successfully created secrets!")
    else:
      with open(self.fp, "r") as f:
        self.secrets = json.load(f)
      logger.debug("Successfully loaded secrets!")

  def __repr__(self):
    return json.dumps(self.secrets, indent=2)

  def get(self, item, default=None):
    # with open(self.fp, "r") as f:
    #   self.secrets = json.load(f)
    return self.secrets.get(item, default)

  def put(self, item, value, persist: bool = False):
    self.secrets[item] = value
    if persist:
      with open(self.fp, "w") as f:
        f.write(repr(self))


def init_secret():
  # add any logic here for creating secrets
  if not U.env.NBOX_NO_AUTH(False):
    return NBXClient()
  return None

secret = init_secret()
