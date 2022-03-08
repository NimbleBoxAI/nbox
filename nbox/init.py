"""
Build systems are an extremely important detail of any software project. When they work,
they can feel like magic: you execute a command, and after a series of potentially
complicated steps, a working binary (or other artifact) is produced! When they don't work,
they can feel like confusing, non-transparent roadblocks that you wish didn't exist. This
is typical for any powerful tool: magic or a headache depending on the day and the task.
"""

import requests
from .auth import secret
from .utils import logger
from .subway import Sub30


def get_stub():
  try:
    import grpc
    from .hyperloop.nbox_ws_pb2_grpc import WSJobServiceStub
  except ImportError as e:
    logger.warn(f"Could not import gRPC commands, some functionality might not work")
    return None

  creds = grpc.access_token_call_credentials(secret.get("access_token"))
  creds = grpc.composite_channel_credentials(grpc.local_channel_credentials(grpc.LocalConnectionType.UDS), creds)
  channel = grpc.secure_channel("unix:///tmp/jobs-ws.sock", creds)

  TIMEOUT = 1

  logger.info(f"Checking connection on channel for {TIMEOUT}s")
  try:
    grpc.channel_ready_future(channel).result(TIMEOUT)
  except grpc.FutureTimeoutError:
    logger.warn(f"gRPC server timeout, some functionality might not work")
    return None

  nbx_stub = WSJobServiceStub(channel)
  return nbx_stub

def create_webserver_subway(version = "v1"):
  _version_specific_url = secret.get("nbx_url") + f"/api/{version}"
  r = nbox_session.get(_version_specific_url + "/openapi.json")
  try:
    r.raise_for_status()
  except Exception as e:
    logger.error(f"Could not connect to webserver at {secret.get('nbx_url')}")
    logger.error(e)
    return None

  return Sub30(_version_specific_url, r.json(), nbox_session)


# common networking items that will be used everywhere
nbox_session = requests.Session()
nbox_session.headers.update({"Authorization": f"Bearer {secret.get('access_token')}"})
nbox_grpc_stub = get_stub()
nbox_ws_v1 = create_webserver_subway("v1")

# add code here to warn user of nbox deprecation -> not sure how to implement this yet
# raise_old_version_warning()


# I am trying to write code that can be used to create permutations
# ex.:
# time < [hourly, daily, weekly, monthly, yearly]
# client < [users, time]
# _internal < [workspace, user, time]
# projects x _internal

# total_projects
# total_deployments
# member_usage
# project_backup
# runtime
# storage_utilisation_of_project
# weekly_breakdown_of_runtime
# utilisation_daily_weekly
# most_used_hw_config
# api_calls_of_deployments
# deployment_by_online_hours
# utilisation_daily_weekly

# what are all the data points that are being generated?
# Build:
#   - 

# worspace, users and other such are "by" variables which are used to group the data
# in general all the business related things fall in this category
# while all the things from engineering side are the actual data:
#         business: by user, workspace, project
#      engineering: build, deploy (serve), jobs
#            build: disk_space, cpu, memory, network, gpu
#   deploy (serve): model, model_properties, networking, compute (flops), memory (storage)
#             jobs: operators
#
