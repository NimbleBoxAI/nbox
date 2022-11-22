"""
``Relic`` is NimbleBox' Object store APIs to connect to your buckets to provide persistant storage beyond instances.
This can support local file storage along with AWS S3, GCP Bucket, Azure Blob Storage, etc.
"""

from nbox.relics.base import BaseStore
from nbox.relics.local import RelicLocal
from nbox.relics.nbx import RelicsNBX
Relics = RelicsNBX