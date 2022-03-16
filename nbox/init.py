"""
Build systems are an extremely important detail of any software project. When they work,
they can feel like magic: you execute a command, and after a series of potentially
complicated steps, a working binary (or other artifact) is produced! When they don't work,
they can feel like confusing, non-transparent roadblocks that you wish didn't exist. This
is typical for any powerful tool: magic or a headache depending on the day and the task.
"""

import grpc
import requests

from .auth import secret
from .utils import logger
from .subway import Sub30
from .hyperloop.nbox_ws_pb2_grpc import WSJobServiceStub

def get_stub() -> WSJobServiceStub:
  token_cred = grpc.access_token_call_credentials(secret.get("access_token"))
  ssl_creds = grpc.ssl_channel_credentials()
  creds = grpc.composite_channel_credentials(ssl_creds, token_cred)
  channel = grpc.secure_channel("grpc.revamp-online.test-2.nimblebox.ai:443", creds)
  stub = WSJobServiceStub(channel)

  TIMEOUT = 6

  logger.info(f"Checking connection on channel for {TIMEOUT}s")
  try:
    grpc.channel_ready_future(channel).result(TIMEOUT)
  except grpc.FutureTimeoutError:
    logger.warn(f"gRPC server timeout, some functionality might not work")
    return None

  logger.info(f"Connected using stub: {stub}")
  return stub

def create_webserver_subway(version = "v1"):
  _version_specific_url = secret.get("nbx_url") + f"/api/{version}"
  r = nbox_session.get(_version_specific_url + "/openapi.json")
  try:
    r.raise_for_status()
  except Exception as e:
    logger.error(f"Could not connect to webserver at {secret.get('nbx_url')}")
    logger.error(e)
    return None
  out = Sub30(_version_specific_url, r.json(), nbox_session)
  logger.debug(f"Connected to webserver at {out}")
  return out


# common networking items that will be used everywhere
nbox_session = requests.Session()
nbox_session.headers.update({"Authorization": f"Bearer {secret.get('access_token')}"})
nbox_grpc_stub: WSJobServiceStub  = get_stub()
nbox_ws_v1: Sub30 = create_webserver_subway("v1")

# add code here to warn user of nbox deprecation -> not sure how to implement this yet
# raise_old_version_warning()
