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


class ConfigString():
  workspace_id = "config.global.workspace_id"
  workspace_name = "config.global.workspace_name"
  cache = "cache"

  def items():
    return [ConfigString.workspace_id, ConfigString.workspace_name, ConfigString.cache]

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
      
      sess = requests.Session()
      sess.headers = {"Authorization": f"Bearer {access_token}"}
      
      # Once we have the access token, we can get the secrets
      _u = f"{nbx_url}/api/v1"
      r = sess.get(_u+"/user/account_details")
      r.raise_for_status()
      try:
        data = r.json()["data"]
        username = data["username"]
        email = data["email"]
      except Exception as e:
        logger.error(f"Could not get the username and email from the response")
        logger.error(f"This should not have happened, please contact NimbleBox support.")
        raise e

      workspace_id = input("Set the default workspace ID (blank means personal): ").strip()
      workspace_details = {"workspace_name": ""}
      
      if workspace_id:
        # fetch some more data about the workspace
        r = sess.get(_u+"/workspace")
        r.raise_for_status()
        try:
          data = r.json()["data"]
          workspace_details = list(filter(lambda x: x["workspace_id"] == workspace_id, data))
          if len(workspace_details) == 0:
            logger.error(f"Could not find the workspace ID: {workspace_id}. Please check the workspace ID and try again.")
            raise Exception("Invalid workspace ID")
          workspace_details = workspace_details[0]
        except Exception as e:
          logger.error(f"Could not get the workspace details from the response")
          logger.error(f"This should not have happened, please contact NimbleBox support.")
          raise e
      else:
        workspace_id = "personal"
      logger.info(f"Setting the default workspace ID to: {workspace_id}")

      # create the objects
      self.secrets = {
        "email": email,
        "access_token": access_token,
        "nbx_url": nbx_url,
        "username": username,

        # config values that can be set by the user for convenience
        ConfigString.workspace_id: workspace_id,
        ConfigString.workspace_name: workspace_details["workspace_name"],

        # now cache the information about this workspace
        ConfigString.cache: {workspace_id: workspace_details},
      }
      # for k,v in self.secrets.items():
      #   print(type(k), k, "::", type(v), v)
      with open(self.fp, "w") as f:
        f.write(repr(self))
      os.makedirs(U.join(U.env.NBOX_HOME_DIR(), ".cache"), exist_ok=True)
      os.makedirs(U.join(U.env.NBOX_HOME_DIR(), "relics"), exist_ok=True)
      logger.info("Successfully created secrets!")
    else:
      with open(self.fp, "r") as f:
        self.secrets = json.load(f)
      logger.debug("Successfully loaded secrets!")

  def __repr__(self):
    return json.dumps(self.secrets, indent=2)

  def get(self, item, default=None, reload: bool = False):
    if reload:
      with open(self.fp, "r") as f:
        self.secrets = json.load(f)
    return self.secrets.get(item, default)

  def put(self, item, value, persist: bool = False):
    self.secrets[item] = value
    if persist:
      with open(self.fp, "w") as f:
        f.write(repr(self))


def init_secret():
  # add any logic here for creating secrets
  if not U.env.NBOX_NO_AUTH(False):
    secret = NBXClient()
    logger.info(f"Current workspace id: {secret.get(ConfigString.workspace_id)} ({secret.get(ConfigString.workspace_name)})")
    return secret
  else:
    logger.info(f"Skipping authentication as NBOX_NO_AUTH is set to True")
  return None

secret = init_secret()
