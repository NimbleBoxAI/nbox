# Auto generated file. DO NOT EDIT.

import os
import requests
from functools import partial

from google.protobuf.timestamp_pb2 import *
from nbox.sublime.proto.relics_pb2 import *
from nbox.sublime.proto.common_pb2 import *
from nbox.sublime.proto.lmao_pb2 import *

from nbox.sublime._yql.rest_pb2 import Echo
from nbox.sublime._yql.common import *

# ------ Stub ------ #

class LMAO_Stub:
  def __init__(self, url: str, session: requests.Session = None):
    self.url = url.rstrip("/")
    self.session = session or requests.Session()
    self.status = partial(call_rpc, sess = self.session, url = f"{url}/")
    self.protos = partial(call_rpc, sess = self.session, url = f"{url}/protos")

  def init_run(self, _InitRunRequest: InitRunRequest) -> Run:
    echo_resp: Echo = call_rpc(
      self.session,
      f"{self.url}/init_run",
      Echo(message = "InitRunRequest", base64_string=message_to_b64(_InitRunRequest), rpc_name = "init_run")
    )
    if echo_resp is None:
      return None

    _Run = Run() # predefine the output proto
    _Run = b64_to_message(echo_resp.base64_string, _Run)
    return _Run

  def update_run_status(self, _Run: Run) -> Acknowledge:
    echo_resp: Echo = call_rpc(
      self.session,
      f"{self.url}/update_run_status",
      Echo(message = "Run", base64_string=message_to_b64(_Run), rpc_name = "update_run_status")
    )
    if echo_resp is None:
      return None

    _Acknowledge = Acknowledge() # predefine the output proto
    _Acknowledge = b64_to_message(echo_resp.base64_string, _Acknowledge)
    return _Acknowledge

  def on_log(self, _RunLog: RunLog) -> Acknowledge:
    echo_resp: Echo = call_rpc(
      self.session,
      f"{self.url}/on_log",
      Echo(message = "RunLog", base64_string=message_to_b64(_RunLog), rpc_name = "on_log")
    )
    if echo_resp is None:
      return None

    _Acknowledge = Acknowledge() # predefine the output proto
    _Acknowledge = b64_to_message(echo_resp.base64_string, _Acknowledge)
    return _Acknowledge

  def on_save(self, _FileList: FileList) -> Acknowledge:
    echo_resp: Echo = call_rpc(
      self.session,
      f"{self.url}/on_save",
      Echo(message = "FileList", base64_string=message_to_b64(_FileList), rpc_name = "on_save")
    )
    if echo_resp is None:
      return None

    _Acknowledge = Acknowledge() # predefine the output proto
    _Acknowledge = b64_to_message(echo_resp.base64_string, _Acknowledge)
    return _Acknowledge

  def on_train_end(self, _Run: Run) -> Acknowledge:
    echo_resp: Echo = call_rpc(
      self.session,
      f"{self.url}/on_train_end",
      Echo(message = "Run", base64_string=message_to_b64(_Run), rpc_name = "on_train_end")
    )
    if echo_resp is None:
      return None

    _Acknowledge = Acknowledge() # predefine the output proto
    _Acknowledge = b64_to_message(echo_resp.base64_string, _Acknowledge)
    return _Acknowledge

  def list_projects(self, _ListProjectsRequest: ListProjectsRequest) -> ListProjectsResponse:
    echo_resp: Echo = call_rpc(
      self.session,
      f"{self.url}/list_projects",
      Echo(message = "ListProjectsRequest", base64_string=message_to_b64(_ListProjectsRequest), rpc_name = "list_projects")
    )
    if echo_resp is None:
      return None

    _ListProjectsResponse = ListProjectsResponse() # predefine the output proto
    _ListProjectsResponse = b64_to_message(echo_resp.base64_string, _ListProjectsResponse)
    return _ListProjectsResponse

  def list_runs(self, _ListRunsRequest: ListRunsRequest) -> ListRunsResponse:
    echo_resp: Echo = call_rpc(
      self.session,
      f"{self.url}/list_runs",
      Echo(message = "ListRunsRequest", base64_string=message_to_b64(_ListRunsRequest), rpc_name = "list_runs")
    )
    if echo_resp is None:
      return None

    _ListRunsResponse = ListRunsResponse() # predefine the output proto
    _ListRunsResponse = b64_to_message(echo_resp.base64_string, _ListRunsResponse)
    return _ListRunsResponse

  def get_run_details(self, _Run: Run) -> Run:
    echo_resp: Echo = call_rpc(
      self.session,
      f"{self.url}/get_run_details",
      Echo(message = "Run", base64_string=message_to_b64(_Run), rpc_name = "get_run_details")
    )
    if echo_resp is None:
      return None

    _Run = Run() # predefine the output proto
    _Run = b64_to_message(echo_resp.base64_string, _Run)
    return _Run

  def get_run_log(self, _RunLogRequest: RunLogRequest) -> RunLog:
    echo_resp: Echo = call_rpc(
      self.session,
      f"{self.url}/get_run_log",
      Echo(message = "RunLogRequest", base64_string=message_to_b64(_RunLogRequest), rpc_name = "get_run_log")
    )
    if echo_resp is None:
      return None

    _RunLog = RunLog() # predefine the output proto
    _RunLog = b64_to_message(echo_resp.base64_string, _RunLog)
    return _RunLog

  def list_files(self, _Run: Run) -> FileList:
    echo_resp: Echo = call_rpc(
      self.session,
      f"{self.url}/list_files",
      Echo(message = "Run", base64_string=message_to_b64(_Run), rpc_name = "list_files")
    )
    if echo_resp is None:
      return None

    _FileList = FileList() # predefine the output proto
    _FileList = b64_to_message(echo_resp.base64_string, _FileList)
    return _FileList

  def init_serving(self, _InitRunRequest: InitRunRequest) -> Serving:
    echo_resp: Echo = call_rpc(
      self.session,
      f"{self.url}/init_serving",
      Echo(message = "InitRunRequest", base64_string=message_to_b64(_InitRunRequest), rpc_name = "init_serving")
    )
    if echo_resp is None:
      return None

    _Serving = Serving() # predefine the output proto
    _Serving = b64_to_message(echo_resp.base64_string, _Serving)
    return _Serving

  def on_serving_log(self, _LogBuffer: LogBuffer) -> Acknowledge:
    echo_resp: Echo = call_rpc(
      self.session,
      f"{self.url}/on_serving_log",
      Echo(message = "LogBuffer", base64_string=message_to_b64(_LogBuffer), rpc_name = "on_serving_log")
    )
    if echo_resp is None:
      return None

    _Acknowledge = Acknowledge() # predefine the output proto
    _Acknowledge = b64_to_message(echo_resp.base64_string, _Acknowledge)
    return _Acknowledge

  def on_serving_end(self, _Serving: Serving) -> Acknowledge:
    echo_resp: Echo = call_rpc(
      self.session,
      f"{self.url}/on_serving_end",
      Echo(message = "Serving", base64_string=message_to_b64(_Serving), rpc_name = "on_serving_end")
    )
    if echo_resp is None:
      return None

    _Acknowledge = Acknowledge() # predefine the output proto
    _Acknowledge = b64_to_message(echo_resp.base64_string, _Acknowledge)
    return _Acknowledge

  def list_deployments(self, _ListDeploymentsRequest: ListDeploymentsRequest) -> ListDeploymentsResponse:
    echo_resp: Echo = call_rpc(
      self.session,
      f"{self.url}/list_deployments",
      Echo(message = "ListDeploymentsRequest", base64_string=message_to_b64(_ListDeploymentsRequest), rpc_name = "list_deployments")
    )
    if echo_resp is None:
      return None

    _ListDeploymentsResponse = ListDeploymentsResponse() # predefine the output proto
    _ListDeploymentsResponse = b64_to_message(echo_resp.base64_string, _ListDeploymentsResponse)
    return _ListDeploymentsResponse

  def list_servings(self, _ListServingsRequest: ListServingsRequest) -> ListServingsResponse:
    echo_resp: Echo = call_rpc(
      self.session,
      f"{self.url}/list_servings",
      Echo(message = "ListServingsRequest", base64_string=message_to_b64(_ListServingsRequest), rpc_name = "list_servings")
    )
    if echo_resp is None:
      return None

    _ListServingsResponse = ListServingsResponse() # predefine the output proto
    _ListServingsResponse = b64_to_message(echo_resp.base64_string, _ListServingsResponse)
    return _ListServingsResponse

  def get_serving_details(self, _Serving: Serving) -> Serving:
    echo_resp: Echo = call_rpc(
      self.session,
      f"{self.url}/get_serving_details",
      Echo(message = "Serving", base64_string=message_to_b64(_Serving), rpc_name = "get_serving_details")
    )
    if echo_resp is None:
      return None

    _Serving = Serving() # predefine the output proto
    _Serving = b64_to_message(echo_resp.base64_string, _Serving)
    return _Serving

  def get_serving_log(self, _Serving: Serving) -> LogBuffer:
    echo_resp: Echo = call_rpc(
      self.session,
      f"{self.url}/get_serving_log",
      Echo(message = "Serving", base64_string=message_to_b64(_Serving), rpc_name = "get_serving_log")
    )
    if echo_resp is None:
      return None

    _LogBuffer = LogBuffer() # predefine the output proto
    _LogBuffer = b64_to_message(echo_resp.base64_string, _LogBuffer)
    return _LogBuffer


# ------ End Stub ------ #