"""
This file loads first and is responsible for setting up all the global networking items.

..

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
from . import version as V

def get_stub() -> WSJobServiceStub:
  """Create a gRPC stub with the NBX Webserver, this will initialise ``nbox_grpc_stub``
  object which is globally accesible as ``nbox.nbox_grpc_stub``. If you find yourself
  using this function, you might want to reconsider your design."""
  token_cred = grpc.access_token_call_credentials(secret.get("access_token"))
  ssl_creds = grpc.ssl_channel_credentials()
  creds = grpc.composite_channel_credentials(ssl_creds, token_cred)
  channel = grpc.secure_channel(secret.get("nbx_url").replace("https://", "dns:/")+":443", creds)
  stub = WSJobServiceStub(channel)
  future = grpc.channel_ready_future(channel)
  future.add_done_callback(lambda _: logger.info(f"NBX gRPC server is ready"))
  logger.info(f"Connected using stub: {stub.__class__.__name__}")
  return stub

def create_webserver_subway(version: str = "v1", session: requests.Session = None) -> Sub30:
  """Create a Subway object for the NBX Webserver. This is a wrapper around the
  OpenAPI spec plublished by NBX Webserver. It loads the JSON object in ``Sub30``
  which allows accesing REST APIs with python "." (dot) notation. If you find yourself
  using this function, you might want to reconsider your design.

  Returns:
    Sub30: A Subway object for the NBX Webserver.
  """
  _version_specific_url = secret.get("nbx_url") + f"/api/{version}"
  session = session if session != None else nbox_session # select correct session
  r = session.get(_version_specific_url + "/openapi.json")
  try:
    r.raise_for_status()
  except Exception as e:
    logger.error(f"Could not connect to webserver at {secret.get('nbx_url')}")
    logger.error(e)
    return None
  out = Sub30(_version_specific_url, r.json(), session)
  logger.info(f"Connected to webserver at {out}")
  return out


# common networking items that will be used everywhere
nbox_session = requests.Session()
nbox_session.headers.update({"Authorization": f"Bearer {secret.get('access_token')}"})
nbox_grpc_stub: WSJobServiceStub  = get_stub()
nbox_ws_v1: Sub30 = create_webserver_subway(version = "v1", session = nbox_session)

# TODO: @yashbonde: raise deprecation warning for version
# raise_old_version_warning(V._major, V._minor, V._patch)
