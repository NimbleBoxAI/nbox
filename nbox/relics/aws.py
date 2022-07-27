import re
from typing import Tuple

from nbox.relics.base import BaseStore

class S3(BaseStore):
  def __init__(self, s3_url):
    import boto3
    self._boto3 = boto3 # alias
    self.client = self._boto3.client('s3')
    self.s3_url = s3_url
    bucket, prefix = self._split_to_bucket_prefix(s3_url)

  def _split_to_bucket_prefix(self, s3_url: str) -> Tuple[str, str]:
    # clean the url
    if not s3_url.startswith("http"):
      s3_url = re.sub(r"^http[s]://", "", s3_url)
    if not s3_url.startswith("s3."):
      raise Exception(f"Invalid S3 URL: {s3_url}")
    return "", "/path"    

  def _put(self, key: str, value: bytes, ow: bool = False) -> None:
    self.client.put_object(
      Bucket=self.s3_url.split('/')[2],
      Key=key,
      Body=value,
      ACL='public-read',
      StorageClass='STANDARD'
    )
