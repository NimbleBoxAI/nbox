# Auto generated file. DO NOT EDIT.

import os
import requests
from functools import partial

from google.protobuf.timestamp_pb2 import *
from nbox.lmao.proto.lmao_v2_pb2 import *

from nbox.sublime._yql.rest_pb2 import Echo
from nbox.sublime._yql.common import *

# ------ Stub ------ #

class LMAO_Stub:
  def __init__(self, url: str, session: requests.Session = None):
    self.url = url.rstrip("/")
    self.session = session or requests.Session()
    self.status = partial(call_rpc, sess = self.session, url = f"{url}/")
    self.protos = partial(call_rpc, sess = self.session, url = f"{url}/protos")

  def init_project(self, _lmao_v2_InitProjectRequest: InitProjectRequest) -> InitProjectResponse:
    echo_resp: Echo = call_rpc(
      self.session,
      f"{self.url}/init_project",
      Echo(message = "InitProjectRequest", base64_string=message_to_b64(_lmao_v2_InitProjectRequest), rpc_name = "init_project")
    )
    if echo_resp is None:
      return None

    _lmao_v2_InitProjectResponse = InitProjectResponse() # predefine the output proto
    _lmao_v2_InitProjectResponse = b64_to_message(echo_resp.base64_string, _lmao_v2_InitProjectResponse)
    return _lmao_v2_InitProjectResponse

  def get_project(self, _lmao_v2_Project: Project) -> Project:
    echo_resp: Echo = call_rpc(
      self.session,
      f"{self.url}/get_project",
      Echo(message = "Project", base64_string=message_to_b64(_lmao_v2_Project), rpc_name = "get_project")
    )
    if echo_resp is None:
      return None

    _lmao_v2_Project = Project() # predefine the output proto
    _lmao_v2_Project = b64_to_message(echo_resp.base64_string, _lmao_v2_Project)
    return _lmao_v2_Project

  def list_projects(self, _lmao_v2_ListProjectsRequest: ListProjectsRequest) -> ListProjectsResponse:
    echo_resp: Echo = call_rpc(
      self.session,
      f"{self.url}/list_projects",
      Echo(message = "ListProjectsRequest", base64_string=message_to_b64(_lmao_v2_ListProjectsRequest), rpc_name = "list_projects")
    )
    if echo_resp is None:
      return None

    _lmao_v2_ListProjectsResponse = ListProjectsResponse() # predefine the output proto
    _lmao_v2_ListProjectsResponse = b64_to_message(echo_resp.base64_string, _lmao_v2_ListProjectsResponse)
    return _lmao_v2_ListProjectsResponse

  def delete_project(self, _lmao_v2_Project: Project) -> Acknowledge:
    echo_resp: Echo = call_rpc(
      self.session,
      f"{self.url}/delete_project",
      Echo(message = "Project", base64_string=message_to_b64(_lmao_v2_Project), rpc_name = "delete_project")
    )
    if echo_resp is None:
      return None

    _lmao_v2_Acknowledge = Acknowledge() # predefine the output proto
    _lmao_v2_Acknowledge = b64_to_message(echo_resp.base64_string, _lmao_v2_Acknowledge)
    return _lmao_v2_Acknowledge

  def init_run(self, _lmao_v2_InitRunRequest: InitRunRequest) -> Run:
    echo_resp: Echo = call_rpc(
      self.session,
      f"{self.url}/init_run",
      Echo(message = "InitRunRequest", base64_string=message_to_b64(_lmao_v2_InitRunRequest), rpc_name = "init_run")
    )
    if echo_resp is None:
      return None

    _lmao_v2_Run = Run() # predefine the output proto
    _lmao_v2_Run = b64_to_message(echo_resp.base64_string, _lmao_v2_Run)
    return _lmao_v2_Run

  def update_run_status(self, _lmao_v2_Run: Run) -> Acknowledge:
    echo_resp: Echo = call_rpc(
      self.session,
      f"{self.url}/update_run_status",
      Echo(message = "Run", base64_string=message_to_b64(_lmao_v2_Run), rpc_name = "update_run_status")
    )
    if echo_resp is None:
      return None

    _lmao_v2_Acknowledge = Acknowledge() # predefine the output proto
    _lmao_v2_Acknowledge = b64_to_message(echo_resp.base64_string, _lmao_v2_Acknowledge)
    return _lmao_v2_Acknowledge

  def on_log(self, _lmao_v2_RunLog: RunLog) -> Acknowledge:
    echo_resp: Echo = call_rpc(
      self.session,
      f"{self.url}/on_log",
      Echo(message = "RunLog", base64_string=message_to_b64(_lmao_v2_RunLog), rpc_name = "on_log")
    )
    if echo_resp is None:
      return None

    _lmao_v2_Acknowledge = Acknowledge() # predefine the output proto
    _lmao_v2_Acknowledge = b64_to_message(echo_resp.base64_string, _lmao_v2_Acknowledge)
    return _lmao_v2_Acknowledge

  def on_train_end(self, _lmao_v2_Run: Run) -> Acknowledge:
    echo_resp: Echo = call_rpc(
      self.session,
      f"{self.url}/on_train_end",
      Echo(message = "Run", base64_string=message_to_b64(_lmao_v2_Run), rpc_name = "on_train_end")
    )
    if echo_resp is None:
      return None

    _lmao_v2_Acknowledge = Acknowledge() # predefine the output proto
    _lmao_v2_Acknowledge = b64_to_message(echo_resp.base64_string, _lmao_v2_Acknowledge)
    return _lmao_v2_Acknowledge

  def list_runs(self, _lmao_v2_ListRunsRequest: ListRunsRequest) -> ListRunsResponse:
    echo_resp: Echo = call_rpc(
      self.session,
      f"{self.url}/list_runs",
      Echo(message = "ListRunsRequest", base64_string=message_to_b64(_lmao_v2_ListRunsRequest), rpc_name = "list_runs")
    )
    if echo_resp is None:
      return None

    _lmao_v2_ListRunsResponse = ListRunsResponse() # predefine the output proto
    _lmao_v2_ListRunsResponse = b64_to_message(echo_resp.base64_string, _lmao_v2_ListRunsResponse)
    return _lmao_v2_ListRunsResponse

  def get_run_details(self, _lmao_v2_Run: Run) -> Run:
    echo_resp: Echo = call_rpc(
      self.session,
      f"{self.url}/get_run_details",
      Echo(message = "Run", base64_string=message_to_b64(_lmao_v2_Run), rpc_name = "get_run_details")
    )
    if echo_resp is None:
      return None

    _lmao_v2_Run = Run() # predefine the output proto
    _lmao_v2_Run = b64_to_message(echo_resp.base64_string, _lmao_v2_Run)
    return _lmao_v2_Run

  def get_run_log(self, _lmao_v2_RunLogRequest: RunLogRequest) -> RunLog:
    echo_resp: Echo = call_rpc(
      self.session,
      f"{self.url}/get_run_log",
      Echo(message = "RunLogRequest", base64_string=message_to_b64(_lmao_v2_RunLogRequest), rpc_name = "get_run_log")
    )
    if echo_resp is None:
      return None

    _lmao_v2_RunLog = RunLog() # predefine the output proto
    _lmao_v2_RunLog = b64_to_message(echo_resp.base64_string, _lmao_v2_RunLog)
    return _lmao_v2_RunLog

  def get_experiment_table(self, _lmao_v2_GetExperimentTableRequest: GetExperimentTableRequest) -> ExperimentTable:
    echo_resp: Echo = call_rpc(
      self.session,
      f"{self.url}/get_experiment_table",
      Echo(message = "GetExperimentTableRequest", base64_string=message_to_b64(_lmao_v2_GetExperimentTableRequest), rpc_name = "get_experiment_table")
    )
    if echo_resp is None:
      return None

    _lmao_v2_ExperimentTable = ExperimentTable() # predefine the output proto
    _lmao_v2_ExperimentTable = b64_to_message(echo_resp.base64_string, _lmao_v2_ExperimentTable)
    return _lmao_v2_ExperimentTable

  def delete_experiment(self, _lmao_v2_Run: Run) -> Acknowledge:
    echo_resp: Echo = call_rpc(
      self.session,
      f"{self.url}/delete_experiment",
      Echo(message = "Run", base64_string=message_to_b64(_lmao_v2_Run), rpc_name = "delete_experiment")
    )
    if echo_resp is None:
      return None

    _lmao_v2_Acknowledge = Acknowledge() # predefine the output proto
    _lmao_v2_Acknowledge = b64_to_message(echo_resp.base64_string, _lmao_v2_Acknowledge)
    return _lmao_v2_Acknowledge

  def init_serving(self, _lmao_v2_InitRunRequest: InitRunRequest) -> Serving:
    echo_resp: Echo = call_rpc(
      self.session,
      f"{self.url}/init_serving",
      Echo(message = "InitRunRequest", base64_string=message_to_b64(_lmao_v2_InitRunRequest), rpc_name = "init_serving")
    )
    if echo_resp is None:
      return None

    _lmao_v2_Serving = Serving() # predefine the output proto
    _lmao_v2_Serving = b64_to_message(echo_resp.base64_string, _lmao_v2_Serving)
    return _lmao_v2_Serving

  def update_serving_status(self, _lmao_v2_Serving: Serving) -> Acknowledge:
    echo_resp: Echo = call_rpc(
      self.session,
      f"{self.url}/update_serving_status",
      Echo(message = "Serving", base64_string=message_to_b64(_lmao_v2_Serving), rpc_name = "update_serving_status")
    )
    if echo_resp is None:
      return None

    _lmao_v2_Acknowledge = Acknowledge() # predefine the output proto
    _lmao_v2_Acknowledge = b64_to_message(echo_resp.base64_string, _lmao_v2_Acknowledge)
    return _lmao_v2_Acknowledge

  def on_serving_log(self, _lmao_v2_RunLog: RunLog) -> Acknowledge:
    echo_resp: Echo = call_rpc(
      self.session,
      f"{self.url}/on_serving_log",
      Echo(message = "RunLog", base64_string=message_to_b64(_lmao_v2_RunLog), rpc_name = "on_serving_log")
    )
    if echo_resp is None:
      return None

    _lmao_v2_Acknowledge = Acknowledge() # predefine the output proto
    _lmao_v2_Acknowledge = b64_to_message(echo_resp.base64_string, _lmao_v2_Acknowledge)
    return _lmao_v2_Acknowledge

  def on_serving_end(self, _lmao_v2_Serving: Serving) -> Acknowledge:
    echo_resp: Echo = call_rpc(
      self.session,
      f"{self.url}/on_serving_end",
      Echo(message = "Serving", base64_string=message_to_b64(_lmao_v2_Serving), rpc_name = "on_serving_end")
    )
    if echo_resp is None:
      return None

    _lmao_v2_Acknowledge = Acknowledge() # predefine the output proto
    _lmao_v2_Acknowledge = b64_to_message(echo_resp.base64_string, _lmao_v2_Acknowledge)
    return _lmao_v2_Acknowledge

  def list_servings(self, _lmao_v2_ListServingsRequest: ListServingsRequest) -> ListServingsResponse:
    echo_resp: Echo = call_rpc(
      self.session,
      f"{self.url}/list_servings",
      Echo(message = "ListServingsRequest", base64_string=message_to_b64(_lmao_v2_ListServingsRequest), rpc_name = "list_servings")
    )
    if echo_resp is None:
      return None

    _lmao_v2_ListServingsResponse = ListServingsResponse() # predefine the output proto
    _lmao_v2_ListServingsResponse = b64_to_message(echo_resp.base64_string, _lmao_v2_ListServingsResponse)
    return _lmao_v2_ListServingsResponse

  def get_serving_details(self, _lmao_v2_Serving: Serving) -> Serving:
    echo_resp: Echo = call_rpc(
      self.session,
      f"{self.url}/get_serving_details",
      Echo(message = "Serving", base64_string=message_to_b64(_lmao_v2_Serving), rpc_name = "get_serving_details")
    )
    if echo_resp is None:
      return None

    _lmao_v2_Serving = Serving() # predefine the output proto
    _lmao_v2_Serving = b64_to_message(echo_resp.base64_string, _lmao_v2_Serving)
    return _lmao_v2_Serving

  def get_serving_log(self, _lmao_v2_ServingLogRequest: ServingLogRequest) -> ServingLogResponse:
    echo_resp: Echo = call_rpc(
      self.session,
      f"{self.url}/get_serving_log",
      Echo(message = "ServingLogRequest", base64_string=message_to_b64(_lmao_v2_ServingLogRequest), rpc_name = "get_serving_log")
    )
    if echo_resp is None:
      return None

    _lmao_v2_ServingLogResponse = ServingLogResponse() # predefine the output proto
    _lmao_v2_ServingLogResponse = b64_to_message(echo_resp.base64_string, _lmao_v2_ServingLogResponse)
    return _lmao_v2_ServingLogResponse

  def delete_serving(self, _lmao_v2_Serving: Serving) -> Acknowledge:
    echo_resp: Echo = call_rpc(
      self.session,
      f"{self.url}/delete_serving",
      Echo(message = "Serving", base64_string=message_to_b64(_lmao_v2_Serving), rpc_name = "delete_serving")
    )
    if echo_resp is None:
      return None

    _lmao_v2_Acknowledge = Acknowledge() # predefine the output proto
    _lmao_v2_Acknowledge = b64_to_message(echo_resp.base64_string, _lmao_v2_Acknowledge)
    return _lmao_v2_Acknowledge

  def get_rule_builder(self, _lmao_v2_RuleBuilder: RuleBuilder) -> RuleBuilder:
    echo_resp: Echo = call_rpc(
      self.session,
      f"{self.url}/get_rule_builder",
      Echo(message = "RuleBuilder", base64_string=message_to_b64(_lmao_v2_RuleBuilder), rpc_name = "get_rule_builder")
    )
    if echo_resp is None:
      return None

    _lmao_v2_RuleBuilder = RuleBuilder() # predefine the output proto
    _lmao_v2_RuleBuilder = b64_to_message(echo_resp.base64_string, _lmao_v2_RuleBuilder)
    return _lmao_v2_RuleBuilder

  def create_rule(self, _lmao_v2_Rule: Rule) -> Rule:
    echo_resp: Echo = call_rpc(
      self.session,
      f"{self.url}/create_rule",
      Echo(message = "Rule", base64_string=message_to_b64(_lmao_v2_Rule), rpc_name = "create_rule")
    )
    if echo_resp is None:
      return None

    _lmao_v2_Rule = Rule() # predefine the output proto
    _lmao_v2_Rule = b64_to_message(echo_resp.base64_string, _lmao_v2_Rule)
    return _lmao_v2_Rule

  def list_rules(self, _lmao_v2_ListRules: ListRules) -> ListRules:
    echo_resp: Echo = call_rpc(
      self.session,
      f"{self.url}/list_rules",
      Echo(message = "ListRules", base64_string=message_to_b64(_lmao_v2_ListRules), rpc_name = "list_rules")
    )
    if echo_resp is None:
      return None

    _lmao_v2_ListRules = ListRules() # predefine the output proto
    _lmao_v2_ListRules = b64_to_message(echo_resp.base64_string, _lmao_v2_ListRules)
    return _lmao_v2_ListRules

  def delete_rule(self, _lmao_v2_Rule: Rule) -> Acknowledge:
    echo_resp: Echo = call_rpc(
      self.session,
      f"{self.url}/delete_rule",
      Echo(message = "Rule", base64_string=message_to_b64(_lmao_v2_Rule), rpc_name = "delete_rule")
    )
    if echo_resp is None:
      return None

    _lmao_v2_Acknowledge = Acknowledge() # predefine the output proto
    _lmao_v2_Acknowledge = b64_to_message(echo_resp.base64_string, _lmao_v2_Acknowledge)
    return _lmao_v2_Acknowledge

  def update_rule(self, _lmao_v2_Rule: Rule) -> Acknowledge:
    echo_resp: Echo = call_rpc(
      self.session,
      f"{self.url}/update_rule",
      Echo(message = "Rule", base64_string=message_to_b64(_lmao_v2_Rule), rpc_name = "update_rule")
    )
    if echo_resp is None:
      return None

    _lmao_v2_Acknowledge = Acknowledge() # predefine the output proto
    _lmao_v2_Acknowledge = b64_to_message(echo_resp.base64_string, _lmao_v2_Acknowledge)
    return _lmao_v2_Acknowledge


# ------ End Stub ------ #