from ..auth import AWSClient

def s3_download(client: AWSClient, bucket_name: str, key: str, local_path: str):
<<<<<<< HEAD
  s3c = client.get_client("s3")
  s3c.download_file(
=======
  client.download_file(
>>>>>>> master
    Bucket = bucket_name,
    Key = key,
    Filename = local_path
  )

__all__ = [
  "s3_download"
]