import os
import json
import logging
import requests
from getpass import getpass

from .utils import join, nbox_session, NBOX_HOME_DIR

logger = logging.getLogger()

# ------ AWS Auth ------ #

class AWSClient:
  def __init__(self, aws_access_key_id, aws_secret_access_key, region_name, **boto_config_kwargs):
    import boto3
    from botocore.client import Config as BotoConfig

    self.aws_access_key_id = aws_access_key_id
    self.aws_secret_access_key = aws_secret_access_key
    self.region_name = region_name

    self.client = boto3.client(
      "s3",
      aws_access_key_id=self.aws_access_key_id,
      aws_secret_access_key=self.aws_secret_access_key,
      region_name=self.region_name,
      config = BotoConfig(
        signature_version="s3v4",
        **boto_config_kwargs
      )
    )

# ------ GCP Auth ------ #

class GCPClient:
  def __init__(self):
    raise NotImplementedError()

# ------ Azure Auth ------ #

class AzureClient:
  def __init__(self):
    raise NotImplementedError()

# ------ OCI Auth ------ #

class OCIClient:
  def __init__(self):
    raise NotImplementedError()


# ------ Digital Ocean Auth ------ #

class DOClient:
  def __init__(self):
    raise NotImplementedError()

# ------ NBX Auth ------ #

class NBXClient:
  @staticmethod
  def get_access_token(nbx_home_url, username, password=None):
    password = getpass("Password: ") if password is None else password
    try:
      r = requests.post(url=f"{nbx_home_url}/api/user/login", json={"username": username, "password": password})
    except Exception as e:
      raise Exception(f"Could not connect to NBX | {str(e)}")

    if r.status_code == 401:
      logger.error(" Invalid username/password. Please try again!")
      return False
    elif r.status_code == 200:
      access_packet = r.json()
      access_token = access_packet.get("access_token", None)
      logger.info("Access token obtained")
      return access_token
    else:
      logger.error(f"Unknown error: {r.content.decode()}")
      raise Exception(f"Unknown error: {r.status_code}")

  @staticmethod
  def create_secret_file(username, access_token, nbx_url):
    os.makedirs(NBOX_HOME_DIR, exist_ok=True)
    fp = join(NBOX_HOME_DIR, "secrets.json")
    with open(fp, "w") as f:
      f.write(json.dumps({"username": username, "access_token": access_token, "nbx_url": nbx_url}, indent=2))

  def __init__(self):
    os.makedirs(NBOX_HOME_DIR, exist_ok=True)
    fp = join(NBOX_HOME_DIR, "secrets.json")

    # if this is the first time starting this then get things from the nbx-hq
    if not os.path.exists(fp):
      # get the secrets JSON
      try:
        self.secrets = json.loads(requests.get(
          "https://raw.githubusercontent.com/NimbleBoxAI/nbox/master/assets/sample_config.json"
          ).content.decode("utf-8")
        )
      except Exception as e:
        raise Exception(f"Could not create secrets file: {e}")

      # populate with the first time things
      nbx_home_url = "https://www.nimblebox.ai"
      username = input("Username: ")
      access_token = None
      while not access_token:
        access_token = self.get_access_token(nbx_home_url, username)
        self.secrets["access_token"] = access_token
      self.secrets["username"] = username
      self.secrets["nbx_url"] = nbx_home_url
      with open(fp, "w") as f:
        f.write(self.__repr__())
      logger.info("Successfully created secrets!")
    else:
      with open(fp, "r") as f:
        self.secrets = json.load(f)
      logger.info("Successfully loaded secrets!")

  def __repr__(self):
    return json.dumps(self.secrets)

  def get(self, item):
    return self.secrets[item]

# function for manual trigger

def init_secret():
  global secret
  secret = NBXClient()
  nbox_session.headers.update({"Authorization": f"Bearer {secret.get('access_token')}"})

secret = None
if not os.getenv("NBOX_NO_AUTH", False):
  init_secret()
