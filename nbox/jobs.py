"""
`nbox.Job` and `nbox.Serve` are wrappers to the NBX-Jobs and NBX-Deploy APIs and contains staticmethods for convinience from the CLI.

* `datetime.now(timezone.utc)` is incorrect, use [this](https://blog.ganssle.io/articles/2019/11/utcnow.html) method.
"""

import os
import sys
import grpc
import tabulate
from typing import Tuple, List, Dict, Any
from functools import lru_cache, partial
from datetime import datetime, timedelta, timezone
from google.protobuf.timestamp_pb2 import Timestamp
from google.protobuf.field_mask_pb2 import FieldMask

import nbox.utils as U
from nbox.auth import secret, AuthConfig, auth_info_pb
from nbox.utils import logger, lo
from nbox.version import __version__
from nbox import messages as mpb
# from nbox.messages import rpc, streaming_rpc
from nbox.init import nbox_grpc_stub, nbox_ws_v1, nbox_serving_service_stub, nbox_model_service_stub
from nbox.nbxlib.astea import Astea, IndexTypes as IT

from nbox.hyperloop.jobs.nbox_ws_pb2 import JobRequest
from nbox.hyperloop.jobs.job_pb2 import Job as JobProto
from nbox.hyperloop.jobs.dag_pb2 import DAG as DAGProto
from nbox.hyperloop.common.common_pb2 import Resource
from nbox.hyperloop.jobs.nbox_ws_pb2 import (
  ListJobsRequest,
  ListJobsResponse,
  UpdateJobRequest
)
from nbox.hyperloop.deploy.serve_pb2 import (
  ServingListResponse,
  ServingRequest,
  Serving,
  ServingListRequest,
  UpdateServingRequest,
  ModelRequest,
  Model as ModelProto,
  UpdateModelRequest,
)


DEFAULT_RESOURCE = Resource(
  cpu = "128m",         # 100mCPU
  memory = "256Mi",     # MiB
  disk_size = "3Gi",    # GiB
  gpu = "none",         # keep "none" for no GPU
  gpu_count = "0",      # keep "0" when no GPU
  timeout = 120_000,    # 2 minutes between attempts
  max_retries = 2,      # third times the charm :P
)

from nbox.jd_core import (
    Schedule,
    _get_job_data,
    print_job_list,
    Job,
    _get_deployment_data,
    print_serving_list,
    Serve,
    JobsCli,
    ServeCli
)
