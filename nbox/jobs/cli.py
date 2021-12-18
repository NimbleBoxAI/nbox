import logging
from typing import List

from .jobs import (
  print_status, get_status, Instance, Subway
)
from ..utils import nbox_session

logger = logging.getLogger("cli")
web_server_subway = Subway("https://nimblebox.ai", nbox_session)

def start(
  name: str,
  loc = None,
  cpu_only: bool = True,
  gpu_count: int = 1,
):
  logger.info(f"Starting {name} on {loc}")
  Instance(name, loc).start(cpu_only, gpu_count)


def stop(
  name: str = "all",
  loc = None
):
  if name == "all":
    logger.info(f"Stopping {all} on {loc}")
    instances_to_stop = []
    money, data = get_status(loc)
    logger.info(f"Money: {money}")
    for item in data:
      if item["state"] == "RUNNING":
        instances_to_stop.append(item)
  else:
    instances_to_stop = [name]

  if not instances_to_stop:
    logger.info("No instances to stop")
    return

  for item in instances_to_stop:
    logger.info(f"Stopping {item['instance_id']}")
    web_server_subway.stop(item["instance_id"])

jobs_cli = {
  "start": start,
  "stop": stop,
}
