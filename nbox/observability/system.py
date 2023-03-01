"""
{% CallOut variant="info" label="This is currently a work in progress, please check back later for progress or reach out to NimbleBox support." /%}
"""

import time
import psutil
import threading
from queue import Queue

from nbox.sublime.proto.lmao_v2_pb2 import *

import GPUtil
HAS_GPU = len(GPUtil.getGPUs())

def get_metrics_dict():
  cpu_usage = psutil.cpu_percent(percpu=True)
  # total_memory = psutil.virtual_memory().total
  data = {
    "cpu_usage": sum(cpu_usage) / len(cpu_usage),
    "memory_available (MB)": psutil.virtual_memory().available // (1024 ** 2),
    "memory_usage (MB)": psutil.virtual_memory().used // (1024 ** 2),
    "memory_percentage": psutil.virtual_memory().percent,
    "disk_utilisation": psutil.disk_usage('/').percent
  }
  if HAS_GPU:
    for i, gpu in enumerate(GPUtil.getGPUs()):
      data.update({
        f"gpu-{i}_usage": gpu.load,
        f"gpu-{i}_memory_available": gpu.memoryFree // (1024 ** 2),
        f"gpu-{i}_memory_usage": gpu.memoryUsed // (1024 ** 2)
      })
  return data


class SystemMetricsLogger:
  def __init__(self, dk, log_every: int = 1) -> None:
    self.dk = dk
    self.log_every = log_every

    # create a rate limiting mechanism
    self._queue = Queue()
    self._bar = threading.Barrier(2)
    def _rate_limiter(s = log_every):
      while True:
        self._bar.wait()
        time.sleep(s)
    self.rl = threading.Thread(target=_rate_limiter, daemon=True)
    self.metrics_logger = threading.Thread(target=self._create_metrics_dict, daemon=True)

    # So this function returns the latest cpu usage thast it has gathered from the previous call
    # and so we need to do an init fire to avoid getting 0.0
    psutil.cpu_percent(percpu=True)

  def start(self):    
    self.rl.start()
    self.metrics_logger.start()

  def _create_metrics_dict(self):
    while True:
      if self.dk.completed:
        break
      data = get_metrics_dict()
      self._queue.put(data)
      run_comp = self.log()
      if run_comp:
        break
      time.sleep(self.log_every)

  def log(self) -> bool:
    if self.dk.completed:
      return True
    self._bar.wait()
    items = []
    while not self._queue.empty():
      items.append(self._queue.get())
    for x in items:
      self.dk.log(x, log_type=RunLog.LogType.SYSTEM)
    return False

  def stop(self):
    self._bar.wait()
    self._bar.wait()
    self.rl.join()
    self.metrics_logger.join()

  def __del__(self):
    self.stop()


if __name__ == "__main__":
  import json
  print(json.dumps(get_metrics_dict(), indent=2))
