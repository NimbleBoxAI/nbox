import re
from typing import Union

from nbox.hyperloop.common.common_pb2 import Resource
from nbox.nbxlib.cloud_machines import AWS, GCP, MSAZ

class PresetMachineConfig(object):
  def __init__(
    self,
    cpu: str = "1000m",
    memory: str = "1Gi",
    gpu: str = "0",
    gpu_name: str = "none",
    disk_size: str = "10Gi",
    metadata: dict = {}
  ) -> None:
    self.cpu = cpu
    self.memory = memory
    self.gpu = gpu
    self.gpu_name = gpu_name
    self.disk_size = disk_size
    self.metadata = metadata

  @property
  def resource(self) -> Resource:
    """Resource object as needed by the NBX Jobs + Deploy Pods"""
    return Resource(
      cpu = self.cpu,
      memory = self.memory,
      gpu_count = self.gpu,
      gpu = self.gpu_name,
      disk_size = self.disk_size,
      max_retries = 2,
      timeout = 120_000,
    )

  def __repr__(self) -> str:
    return str(self.resource)


def get_resource_by_name(name: str) -> Resource:
  if not re.match(r"^[\w]+\.[\w]+$", name):
    raise ValueError(f"Invalid resource name: {name}")
  
  class_name, machine = name.lower().split(".")
  if class_name == "aws":
    cfg = getattr(AWS, machine, None)
  elif class_name == "gcp":
    cfg = getattr(GCP, machine, None)
  elif class_name == "msaz":
    cfg = getattr(MSAZ, machine, None)
  else:
    raise ValueError(f"Invalid resource name: {name}, cloud should be one of: aws, gcp, msaz")
  
  if cfg is None:
    raise ValueError(f"Invalid resource name: {name}")
  return cfg.resource
