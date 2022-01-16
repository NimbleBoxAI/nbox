import os
import logging

def reset_log():
  JSON_LOG = os.environ.get("NBOX_JSON_LOG", False)
  json_format = '{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}'
  normal_format = '[%(asctime)s] [%(levelname)s] %(message)s'

  logging.basicConfig(
    level = logging.INFO,
    format = json_format if JSON_LOG else normal_format,
    datefmt = "%Y-%m-%dT%H:%M:%S%z" # isoformat
  )
