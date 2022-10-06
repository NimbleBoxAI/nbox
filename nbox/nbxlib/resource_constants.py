from typing import Union
from nbox.hyperloop.job_pb2 import Resource

class PresetMachineConfig(object):
  def __init__(
    self,
    cpu: str = "1000m",
    memory: str = "1Gi",
    gpu: str = "0",
    gpu_name: str = "none",
    disk_size: str = "10Gi",
  ) -> None:
    self.cpu = cpu
    self.memory = memory
    self.gpu = gpu
    self.gpu_name = gpu_name
    self.disk_size = disk_size

  def set_gpu(self, gpu: Union[str, int], gpu_name: str) -> None:
    if isinstance(gpu, int):
      assert gpu < 10, "gpu is a string, max 9"
      gpu = str(gpu)
    else:
      assert len(gpu) < 2, "gpu is a string, max 9"

    self.gpu = gpu
    self.gpu_name = gpu_name

  def set_disk_size(self, disk_size: Union[str, int]) -> None:
    max_size = 256
    if isinstance(disk_size, int):
      if disk_size > max_size:
        raise ValueError(f"disk_size is in GiB, max {max_size}")
      disk_size = f"{disk_size}Gi"
    else:
      assert disk_size.endswith("Gi")
      assert int(disk_size[:-2]) <= max_size, f"disk_size is in GiB, max {max_size}"
    self.disk_size = disk_size

  @property
  def resource(self) -> Resource:
    return Resource(
      cpu = self.cpu,
      memory = self.memory,
      gpu = self.gpu,
      gpu_name = self.gpu_name,
      disk_size = self.disk_size,
      max_retries = 2,
      timeout = 120_000,
    )

  def __call__(
    self,
    disk_size: str = None,
    gpu: Union[str, int] = None,
    gpu_name: str = None,
  ) -> 'PresetMachineConfig':
    self.disk_size = disk_size or self.disk_size
    self.gpu = gpu or self.gpu
    self.gpu_name = gpu_name or self.gpu_name


# define all the object below, that user can import directly.

CPU_1_RAM_1 = PresetMachineConfig(cpu = "1000m", memory = "1Gi")
CPU_1_RAM_2 = PresetMachineConfig(cpu = "1000m", memory = "2Gi")
CPU_1_RAM_4 = PresetMachineConfig(cpu = "1000m", memory = "4Gi")
CPU_1_RAM_8 = PresetMachineConfig(cpu = "1000m", memory = "8Gi")
CPU_1_RAM_16 = PresetMachineConfig(cpu = "1000m", memory = "16Gi")
CPU_1_RAM_32 = PresetMachineConfig(cpu = "1000m", memory = "32Gi")
CPU_1_RAM_64 = PresetMachineConfig(cpu = "1000m", memory = "64Gi")

CPU_2_RAM_2 = PresetMachineConfig(cpu = "2000m", memory = "2Gi")
CPU_2_RAM_4 = PresetMachineConfig(cpu = "2000m", memory = "4Gi")
CPU_2_RAM_8 = PresetMachineConfig(cpu = "2000m", memory = "8Gi")
CPU_2_RAM_16 = PresetMachineConfig(cpu = "2000m", memory = "16Gi")
CPU_2_RAM_32 = PresetMachineConfig(cpu = "2000m", memory = "32Gi")
CPU_2_RAM_64 = PresetMachineConfig(cpu = "2000m", memory = "64Gi")

CPU_4_RAM_4 = PresetMachineConfig(cpu = "4000m", memory = "4Gi")
CPU_4_RAM_8 = PresetMachineConfig(cpu = "4000m", memory = "8Gi")
CPU_4_RAM_16 = PresetMachineConfig(cpu = "4000m", memory = "16Gi")
CPU_4_RAM_32 = PresetMachineConfig(cpu = "4000m", memory = "32Gi")
CPU_4_RAM_64 = PresetMachineConfig(cpu = "4000m", memory = "64Gi")

CPU_8_RAM_8 = PresetMachineConfig(cpu = "8000m", memory = "8Gi")
CPU_8_RAM_16 = PresetMachineConfig(cpu = "8000m", memory = "16Gi")
CPU_8_RAM_32 = PresetMachineConfig(cpu = "8000m", memory = "32Gi")
CPU_8_RAM_64 = PresetMachineConfig(cpu = "8000m", memory = "64Gi")

CPU_16_RAM_16 = PresetMachineConfig(cpu = "16000m", memory = "16Gi")
CPU_16_RAM_32 = PresetMachineConfig(cpu = "16000m", memory = "32Gi")
CPU_16_RAM_64 = PresetMachineConfig(cpu = "16000m", memory = "64Gi")

CPU_32_RAM_32 = PresetMachineConfig(cpu = "32000m", memory = "32Gi")
CPU_32_RAM_64 = PresetMachineConfig(cpu = "32000m", memory = "64Gi")

CPU_64_RAM_64 = PresetMachineConfig(cpu = "64000m", memory = "64Gi")

def get_resource_by_name(name: str) -> PresetMachineConfig:
  res = globals().get(name, None)
  if res is None:
    raise ValueError(f"Resource {name} not found")
  return res


# when I have to disk_size

machine = CPU_16_RAM_32
machine.set_disk_size(100)
machine.resource

machine = CPU_16_RAM_32(disk_size=100)
machine.resource

# when I have to add GPU

machine = CPU_16_RAM_32
machine.set_gpu(gpu=1, gpu_name="nvidia-tesla-v100")
machine.resource

machine = CPU_16_RAM_32(gpu=1, gpu_name="nvidia-tesla-v100")
machine.resource

