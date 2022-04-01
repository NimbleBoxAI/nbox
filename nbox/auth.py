"""
This code is used to manage the authentication of the entire ``nbox`` package. For authentication
it will create a ``.nbx`` in the user's home directory (``~/.nbx``, in case of linux) and store
a file called ``secrets.json``. This folder will also contain more information and files that
are used elsewhere as well ex. files generated when takling to any instance.

We have also provided simple built in methods to connect to your cloud provider service as a part
of BYOC (bring your own cloud), they are still work on progress, of you are interested in them,
please raise an issue on Github.

"""
import os
import json
import requests
import webbrowser
from getpass import getpass

from .utils import join, NBOX_HOME_DIR, isthere, logger
from .subway import Subway

# ------ AWS Auth ------ #

class AWSClient:
  @isthere("boto3", "botocore", soft = False)
  def __init__(self, aws_access_key_id: str, aws_secret_access_key: str, region_name: str):
    """Template for creating your own AWS authentication class.

    EXPERIMENTAL: This is not yet ready for use.

    Args:
        aws_access_key_id (str): AWS access key ID
        aws_secret_access_key (str): AWS secret access key
        region_name (str): AWS region name
    """
    self.aws_access_key_id = aws_access_key_id
    self.aws_secret_access_key = aws_secret_access_key
    self.region_name = region_name

  def get_client(self, service_name: str = "s3", **boto_config_kwargs):
    """Get the client object for the given service

    Args:
        service_name (str): _description_. Defaults to "s3".
    """
    import boto3
    from botocore.client import Config as BotoConfig

    return boto3.client(
      service_name,
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
  @isthere("google-cloud-sdk", "google-cloud-storage", soft = False)
  def __init__(self, project_id: str, credentials_file):
    """Template for creating your own authentication class.

    EXPERIMENTAL: This is not yet ready for use.

    Args:
        project_id (str): GCP project ID
        credentials_file: GCP credentials python object, must support .read() method
    """

    from google.oauth2 import service_account
    
    self.project_id = project_id
    self.credentials_file = credentials_file
    self.creds = service_account.Credentials.from_service_account_file(
      self.credentials_file,
      scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )

  def get_client(self, service_name: str = "storage", **gcp_config_kwargs):
    """Get the client object for the given service

    Args:
        service_name (str): GCP service name
    """
    if service_name == "storage":
      from google.cloud import storage
      return storage.Client(
        project=self.project_id,
        credentials=self.creds,
        **gcp_config_kwargs
      )
    

# ------ Azure Auth ------ #

class AzureClient:
  @isthere("azure-storage-blob", soft = False)
  def __init__(self):
    """
    Microsoft Azure

    EXPERIMENTAL: This is not yet ready for use.
    """
    from azure.storage.blob import BlobServiceClient
    from azure.identity import DefaultAzureCredential

    self.blob_service_client = BlobServiceClient(
      credential=DefaultAzureCredential(),
      endpoint="https://nbox.blob.core.windows.net"
    )

  def get_client(self, service_name = "blob", **azure_config_kwargs):
    if service_name == "blob":
      from azure.storage.blob import BlobClient

      return BlobClient(
        self.blob_service_client,
        **azure_config_kwargs
      )

# ------ OCI Auth ------ #

class OCIClient:
  @isthere("oci", "oci-py", soft = False)
  def __init__(self, config_file):
    """
    Oracle Cloud Infrastructure

    EXPERIMENTAL: This is not yet ready for use.
    """
    from oci.config import from_file
    from oci.signer import Signer

    self.config = from_file(config_file)
    self.signer = Signer(
      tenancy=self.config["tenancy"],
      user=self.config["user"],
      fingerprint=self.config["fingerprint"],
      private_key_file_location=self.config["key_file"]
    )

  def get_client(self, service_name = "object_storage", **oci_config_kwargs):

    if service_name == "object_storage":
      from oci.object_storage.models import CreateBucketDetails
      from oci.object_storage.models import CreateMultipartUploadDetails
      from oci.object_storage.models import Object
      from oci.object_storage.models import UploadPartDetails
      from oci.object_storage.object_storage_client import ObjectStorageClient

      return ObjectStorageClient(
        self.config["user"],
        self.signer,
        **oci_config_kwargs
      )


# ------ Digital Ocean Auth ------ #

class DOClient:
  @isthere("doctl", soft = False)
  def __init__(self, config_file):
    """
    Digital Ocean

    EXPERIMENTAL: This is not yet ready for use.
    """
    from doctl.doctl_client import DictCursor
    from doctl.doctl_client import DoctlClient

    self.doctl_client = DoctlClient(
      config_file=config_file,
      cursor_class=DictCursor
    )
  
  def get_client(self, service_name = "object_storage", **oci_config_kwargs):
    if service_name == "object_storage":
      from doctl.object_storage.object_storage_client import ObjectStorageClient

      return ObjectStorageClient(
        self.doctl_client,
        **oci_config_kwargs
      )

# ------ NBX Auth ------ #

class NBXClient:
  def __init__(self, nbx_url = "https://app.nimblebox.ai"):
    """We try to find the values secrets file in the ``~/.nbx/secrets.json``, if not
    found, we ask the user for the email and direct them to browser.
    """
    os.makedirs(NBOX_HOME_DIR, exist_ok=True)
    fp = join(NBOX_HOME_DIR, "secrets.json")

    # if this is the first time starting this then get things from the nbx-hq
    if not os.path.exists(fp):
      logger.info(f"Ensure that you put the email ID you have signed up with!")
      email_id = input("Email ID: ")
      _secrets_url = f"{nbx_url}/secrets"
      logger.info(f"Opening: {_secrets_url}")
      webbrowser.open(_secrets_url)
      access_token = getpass("Access Token: ")
      self.secrets = {
        "email_id": email_id,
        "access_token": access_token,
        "nbx_url": nbx_url
      }
      with open(fp, "w") as f:
        f.write(repr(self))
      logger.info("Successfully created secrets!")
    else:
      with open(fp, "r") as f:
        self.secrets = json.load(f)
      logger.debug("Successfully loaded secrets!")

  def __repr__(self):
    return json.dumps(self.secrets)

  def get(self, item, default=None):
    return self.secrets.get(item, default)

# function for manual trigger
def init_secret():
  # add any logic here for creating secrets
  return NBXClient()

secret = init_secret()
