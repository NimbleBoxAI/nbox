from ..auth import AWSClient

def s3_download(client: AWSClient, bucket_name: str, key: str, local_path: str):
  client.download_file(
    Bucket = bucket_name,
    Key = key,
    Filename = local_path
  )

__all__ = [
  "s3_download"
]