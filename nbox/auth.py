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
from typing import Dict
from getpass import getpass
from functools import lru_cache

import nbox.utils as U
from nbox.utils import join, logger


class AuthConfig():
  workspace_id = "config.global.workspace_id"
  workspace_name = "config.global.workspace_name"
  _workspace_id = "workspace_id"
  email = "email"
  cache = "cache"
  username = "username"
  access_token = "access_token"
  url = "nbx_url"

  # things for the pod
  nbx_pod_run = "run"
  # nbx_pod_deploy = "deploy"

  def items():
    return [AuthConfig.workspace_id, AuthConfig.workspace_name, AuthConfig.cache]

# kept for legacy reasons, remove it when this code (see git blame) is > 4 months old
ConfigString = AuthConfig


class JobDetails(object):
  job_id: str
  run_id: str

# class DeployDetails(object):

class NBXClient:
  def __init__(self, nbx_url = "https://app.nimblebox.ai"):
    """
    Single source for all the secrets file. Can persist across multiple processes.
    """
    os.makedirs(U.env.NBOX_HOME_DIR(), exist_ok=True)
    self.fp = join(U.env.NBOX_HOME_DIR(), "secrets.json")

    access_token = U.env.NBOX_ACCESS_TOKEN("")

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
        AuthConfig.email: email,
        AuthConfig.access_token: access_token,
        AuthConfig.url: nbx_url,
        AuthConfig.username: username,

        # config values that can be set by the user for convenience
        AuthConfig.workspace_id: workspace_id,
        AuthConfig.workspace_name: workspace_details["workspace_name"],

        # now cache the information about this workspace
        AuthConfig.cache: {workspace_id: {
          "workspace_name": workspace_details["workspace_name"],
          "workspace_id": workspace_id,
        }},
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
    """
    Get the value of the item from the secrets file.

    Args:
      item (str): The item to get the value for
      default (any): The default value to return if the item is not found
      reload (bool): If True, reload the secrets file before getting the value

    Returns:
      any: The value of the item or the default value if the item is not found
    """
    if reload:
      with open(self.fp, "r") as f:
        self.secrets = json.load(f)
    return self.secrets.get(item, default)

  def put(self, item, value = None, persist: bool = False):
    """
    Put the value of the item in the secrets file. If no `value` or `persist` is specified,
    this is a no-op.

    Args:
      item (str): The item to put the value for
      value (any): The value to put
      persist (bool): If True, persist the secrets file after putting the value
    """
    if value:
      self.secrets[item] = value
    if persist:
      with open(self.fp, "w") as f:
        f.write(repr(self))

  def __call__(self, item, default=None, reload: bool = False):
    return self.get(item, default, reload)

  @property
  def workspace_id(self) -> str:
    return self.get(AuthConfig.workspace_id) or self.get(AuthConfig._workspace_id)

  @property
  def nbx_url(self) -> str:
    return self.get(AuthConfig.url)

  @property
  def access_token(self) -> str:
    return self.get(AuthConfig.access_token)
  
  @property
  def username(self) -> str:
    return self.get(AuthConfig.username)

  def get_agent_details(self) -> Dict[str, str]:
    if ConfigString.nbx_pod_run in self.secrets:
      run_data = self.secrets[ConfigString.nbx_pod_run]
      jd = JobDetails()
      jd.job_id = run_data.get("job_id", None)
      jd.run_id = run_data.get("token", None)
      return jd
    # elif ConfigString.nbx_pod_deploy in self.secrets:
    #   return self.secrets[ConfigString.nbx_pod_deploy]
    return {}


def init_secret():
  """
  Initialize the secret object. This is a singleton object that can be used across multiple processes.
  """
  # add any logic here for creating secrets
  if not U.env.NBOX_NO_AUTH(False):
    secret = NBXClient()
    logger.info(f"Current workspace id: {secret(AuthConfig.workspace_id)} ({secret(AuthConfig.workspace_name)})")
    return secret
  else:
    logger.info(f"Skipping authentication as NBOX_NO_AUTH is set to True")
  return None

secret = init_secret()


@lru_cache()
def auth_info_pb():
  """
  Get the auth token for the current user.
  """
  from nbox.hyperloop.common.common_pb2 import NBXAuthInfo

  return NBXAuthInfo(
    username = secret(AuthConfig.username),
    workspace_id = secret(AuthConfig.workspace_id) or secret(AuthConfig._workspace_id),
    access_token = secret(AuthConfig.access_token),
  )

def inside_pod():
  return secret(AuthConfig.nbx_pod_run, False)
