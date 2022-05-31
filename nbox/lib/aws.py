from nbox import Operator

class S3Operator(Operator):
  def __init__(self, bucket: str, key: str, local_path: str, mode: str = "upload"):
    """Upload or download a file from S3"""
    from pathlib import Path
    super().__init__()
    self.bucket = bucket
    self.key = key
    self.local_path = Path(local_path)
    self.mode = mode

    if self.mode not in ["upload", "download"]:
      raise Exception(f"Unknown mode: {self.mode}")

  def forward(self):
    import boto3
    s3 = boto3.client("s3")
    if self.mode == "upload":
      s3.upload_file(self.local_path, self.bucket, self.key)
    elif self.mode == "download":
      s3.download_file(self.bucket, self.key, self.local_path)
    else:
      raise Exception("Unknown mode")

