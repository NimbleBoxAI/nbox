# Auto generated file. DO NOT EDIT.

import os
import requests
from functools import partial

from google.protobuf.timestamp_pb2 import *
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

  def init_run(self, _lmao_InitRunRequest: InitRunRequest) -> Run:
    echo_resp: Echo = call_rpc(
      self.session,
      f"{self.url}/init_run",
      Echo(message = "InitRunRequest", base64_string=message_to_b64(_lmao_InitRunRequest), rpc_name = "init_run")
    )
    if echo_resp is None:
      return None

    _lmao_Run = Run() # predefine the output proto
    _lmao_Run = b64_to_message(echo_resp.base64_string, _lmao_Run)
    return _lmao_Run

  def update_run_status(self, _lmao_Run: Run) -> Acknowledge:
    echo_resp: Echo = call_rpc(
      self.session,
      f"{self.url}/update_run_status",
      Echo(message = "Run", base64_string=message_to_b64(_lmao_Run), rpc_name = "update_run_status")
    )
    if echo_resp is None:
      return None

    _lmao_Acknowledge = Acknowledge() # predefine the output proto
    _lmao_Acknowledge = b64_to_message(echo_resp.base64_string, _lmao_Acknowledge)
    return _lmao_Acknowledge

  def on_log(self, _lmao_RunLog: RunLog) -> Acknowledge:
    echo_resp: Echo = call_rpc(
      self.session,
      f"{self.url}/on_log",
      Echo(message = "RunLog", base64_string=message_to_b64(_lmao_RunLog), rpc_name = "on_log")
    )
    if echo_resp is None:
      return None

    _lmao_Acknowledge = Acknowledge() # predefine the output proto
    _lmao_Acknowledge = b64_to_message(echo_resp.base64_string, _lmao_Acknowledge)
    return _lmao_Acknowledge

  def on_save(self, _lmao_FileList: FileList) -> Acknowledge:
    echo_resp: Echo = call_rpc(
      self.session,
      f"{self.url}/on_save",
      Echo(message = "FileList", base64_string=message_to_b64(_lmao_FileList), rpc_name = "on_save")
    )
    if echo_resp is None:
      return None

    _lmao_Acknowledge = Acknowledge() # predefine the output proto
    _lmao_Acknowledge = b64_to_message(echo_resp.base64_string, _lmao_Acknowledge)
    return _lmao_Acknowledge

  def on_train_end(self, _lmao_Run: Run) -> Acknowledge:
    echo_resp: Echo = call_rpc(
      self.session,
      f"{self.url}/on_train_end",
      Echo(message = "Run", base64_string=message_to_b64(_lmao_Run), rpc_name = "on_train_end")
    )
    if echo_resp is None:
      return None

    _lmao_Acknowledge = Acknowledge() # predefine the output proto
    _lmao_Acknowledge = b64_to_message(echo_resp.base64_string, _lmao_Acknowledge)
    return _lmao_Acknowledge

  def list_projects(self, _lmao_ListProjectsRequest: ListProjectsRequest) -> ListProjectsResponse:
    echo_resp: Echo = call_rpc(
      self.session,
      f"{self.url}/list_projects",
      Echo(message = "ListProjectsRequest", base64_string=message_to_b64(_lmao_ListProjectsRequest), rpc_name = "list_projects")
    )
    if echo_resp is None:
      return None

    _lmao_ListProjectsResponse = ListProjectsResponse() # predefine the output proto
    _lmao_ListProjectsResponse = b64_to_message(echo_resp.base64_string, _lmao_ListProjectsResponse)
    return _lmao_ListProjectsResponse

  def list_runs(self, _lmao_ListRunsRequest: ListRunsRequest) -> ListRunsResponse:
    echo_resp: Echo = call_rpc(
      self.session,
      f"{self.url}/list_runs",
      Echo(message = "ListRunsRequest", base64_string=message_to_b64(_lmao_ListRunsRequest), rpc_name = "list_runs")
    )
    if echo_resp is None:
      return None

    _lmao_ListRunsResponse = ListRunsResponse() # predefine the output proto
    _lmao_ListRunsResponse = b64_to_message(echo_resp.base64_string, _lmao_ListRunsResponse)
    return _lmao_ListRunsResponse

  def get_experiment_table(self, _lmao_GetExperimentTableRequest: GetExperimentTableRequest) -> ExperimentTable:
    echo_resp: Echo = call_rpc(
      self.session,
      f"{self.url}/get_experiment_table",
      Echo(message = "GetExperimentTableRequest", base64_string=message_to_b64(_lmao_GetExperimentTableRequest), rpc_name = "get_experiment_table")
    )
    if echo_resp is None:
      return None

    _lmao_ExperimentTable = ExperimentTable() # predefine the output proto
    _lmao_ExperimentTable = b64_to_message(echo_resp.base64_string, _lmao_ExperimentTable)
    return _lmao_ExperimentTable

  def get_run_details(self, _lmao_Run: Run) -> Run:
    echo_resp: Echo = call_rpc(
      self.session,
      f"{self.url}/get_run_details",
      Echo(message = "Run", base64_string=message_to_b64(_lmao_Run), rpc_name = "get_run_details")
    )
    if echo_resp is None:
      return None

    _lmao_Run = Run() # predefine the output proto
    _lmao_Run = b64_to_message(echo_resp.base64_string, _lmao_Run)
    return _lmao_Run

  def get_run_log(self, _lmao_RunLogRequest: RunLogRequest) -> RunLog:
    echo_resp: Echo = call_rpc(
      self.session,
      f"{self.url}/get_run_log",
      Echo(message = "RunLogRequest", base64_string=message_to_b64(_lmao_RunLogRequest), rpc_name = "get_run_log")
    )
    if echo_resp is None:
      return None

    _lmao_RunLog = RunLog() # predefine the output proto
    _lmao_RunLog = b64_to_message(echo_resp.base64_string, _lmao_RunLog)
    return _lmao_RunLog

  def list_files(self, _lmao_Run: Run) -> FileList:
    echo_resp: Echo = call_rpc(
      self.session,
      f"{self.url}/list_files",
      Echo(message = "Run", base64_string=message_to_b64(_lmao_Run), rpc_name = "list_files")
    )
    if echo_resp is None:
      return None

    _lmao_FileList = FileList() # predefine the output proto
    _lmao_FileList = b64_to_message(echo_resp.base64_string, _lmao_FileList)
    return _lmao_FileList

  def delete_experiment(self, _lmao_Run: Run) -> Acknowledge:
    echo_resp: Echo = call_rpc(
      self.session,
      f"{self.url}/delete_experiment",
      Echo(message = "Run", base64_string=message_to_b64(_lmao_Run), rpc_name = "delete_experiment")
    )
    if echo_resp is None:
      return None

    _lmao_Acknowledge = Acknowledge() # predefine the output proto
    _lmao_Acknowledge = b64_to_message(echo_resp.base64_string, _lmao_Acknowledge)
    return _lmao_Acknowledge

  def init_serving(self, _lmao_InitRunRequest: InitRunRequest) -> Serving:
    echo_resp: Echo = call_rpc(
      self.session,
      f"{self.url}/init_serving",
      Echo(message = "InitRunRequest", base64_string=message_to_b64(_lmao_InitRunRequest), rpc_name = "init_serving")
    )
    if echo_resp is None:
      return None

    _lmao_Serving = Serving() # predefine the output proto
    _lmao_Serving = b64_to_message(echo_resp.base64_string, _lmao_Serving)
    return _lmao_Serving

  def on_serving_log(self, _lmao_LogBuffer: LogBuffer) -> Acknowledge:
    echo_resp: Echo = call_rpc(
      self.session,
      f"{self.url}/on_serving_log",
      Echo(message = "LogBuffer", base64_string=message_to_b64(_lmao_LogBuffer), rpc_name = "on_serving_log")
    )
    if echo_resp is None:
      return None

    _lmao_Acknowledge = Acknowledge() # predefine the output proto
    _lmao_Acknowledge = b64_to_message(echo_resp.base64_string, _lmao_Acknowledge)
    return _lmao_Acknowledge

  def on_serving_end(self, _lmao_Serving: Serving) -> Acknowledge:
    echo_resp: Echo = call_rpc(
      self.session,
      f"{self.url}/on_serving_end",
      Echo(message = "Serving", base64_string=message_to_b64(_lmao_Serving), rpc_name = "on_serving_end")
    )
    if echo_resp is None:
      return None

    _lmao_Acknowledge = Acknowledge() # predefine the output proto
    _lmao_Acknowledge = b64_to_message(echo_resp.base64_string, _lmao_Acknowledge)
    return _lmao_Acknowledge

  def list_deployments(self, _lmao_ListDeploymentsRequest: ListDeploymentsRequest) -> ListDeploymentsResponse:
    echo_resp: Echo = call_rpc(
      self.session,
      f"{self.url}/list_deployments",
      Echo(message = "ListDeploymentsRequest", base64_string=message_to_b64(_lmao_ListDeploymentsRequest), rpc_name = "list_deployments")
    )
    if echo_resp is None:
      return None

    _lmao_ListDeploymentsResponse = ListDeploymentsResponse() # predefine the output proto
    _lmao_ListDeploymentsResponse = b64_to_message(echo_resp.base64_string, _lmao_ListDeploymentsResponse)
    return _lmao_ListDeploymentsResponse

  def list_servings(self, _lmao_ListServingsRequest: ListServingsRequest) -> ListServingsResponse:
    echo_resp: Echo = call_rpc(
      self.session,
      f"{self.url}/list_servings",
      Echo(message = "ListServingsRequest", base64_string=message_to_b64(_lmao_ListServingsRequest), rpc_name = "list_servings")
    )
    if echo_resp is None:
      return None

    _lmao_ListServingsResponse = ListServingsResponse() # predefine the output proto
    _lmao_ListServingsResponse = b64_to_message(echo_resp.base64_string, _lmao_ListServingsResponse)
    return _lmao_ListServingsResponse

  def get_serving_details(self, _lmao_Serving: Serving) -> Serving:
    echo_resp: Echo = call_rpc(
      self.session,
      f"{self.url}/get_serving_details",
      Echo(message = "Serving", base64_string=message_to_b64(_lmao_Serving), rpc_name = "get_serving_details")
    )
    if echo_resp is None:
      return None

    _lmao_Serving = Serving() # predefine the output proto
    _lmao_Serving = b64_to_message(echo_resp.base64_string, _lmao_Serving)
    return _lmao_Serving

  def get_serving_log(self, _lmao_Serving: Serving) -> LogBuffer:
    echo_resp: Echo = call_rpc(
      self.session,
      f"{self.url}/get_serving_log",
      Echo(message = "Serving", base64_string=message_to_b64(_lmao_Serving), rpc_name = "get_serving_log")
    )
    if echo_resp is None:
      return None

    _lmao_LogBuffer = LogBuffer() # predefine the output proto
    _lmao_LogBuffer = b64_to_message(echo_resp.base64_string, _lmao_LogBuffer)
    return _lmao_LogBuffer


# ------ End Stub ------ #