import requests
from .auth import secret
from .utils import logger


def get_stub():
  try:
    import grpc
    from .hyperloop.nbox_ws_pb2_grpc import WSJobServiceStub
  except ImportError as e:
    logger.warn(f"Could not import gRPC commands, some functionality might not work")
    return None

  nbx_stub = WSJobServiceStub(
    grpc.secure_channel(
      "grpc.revamp-online.test-2.nimblebox.ai:443",
      grpc.composite_channel_credentials(
        grpc.ssl_channel_credentials(),
        grpc.access_token_call_credentials(secret.get("access_token"))
      )
    )
  )
  return nbx_stub


# common networking items that will be used everywhere
nbox_session = requests.Session()
nbox_session.headers.update({"Authorization": f"Bearer {secret.get('access_token')}"})
nbox_grpc_stub = get_stub()

# add code here to warn user of nbox deprecation -> not sure how to implement this yet
# raise_old_version_warning()
