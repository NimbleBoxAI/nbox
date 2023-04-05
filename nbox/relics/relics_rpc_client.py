# Auto generated file. DO NOT EDIT.

import os
import requests
from functools import partial

from nbox.relics.proto.relics_pb2 import *
from nbox.relics.proto.common_pb2 import *
from nbox.relics.proto.relics_rpc_pb2 import *

from nbox.sublime._yql.rest_pb2 import Echo
from nbox.sublime._yql.common import *

# ------ Stub ------ #

class RelicStore_Stub:
  def __init__(self, url: str, session: requests.Session = None):
    self.url = url.rstrip("/")
    self.session = session or requests.Session()
    self.status = partial(call_rpc, sess = self.session, url = f"{url}/")
    self.protos = partial(call_rpc, sess = self.session, url = f"{url}/protos")

  def create_relic(self, _CreateRelicRequest: CreateRelicRequest) -> Relic:
    echo_resp: Echo = call_rpc(
      self.session,
      f"{self.url}/create_relic",
      Echo(message = "CreateRelicRequest", base64_string=message_to_b64(_CreateRelicRequest), rpc_name = "create_relic")
    )
    if echo_resp is None:
      return None

    _Relic = Relic() # predefine the output proto
    _Relic = b64_to_message(echo_resp.base64_string, _Relic)
    return _Relic

  def list_relics(self, _ListRelicsRequest: ListRelicsRequest) -> ListRelicsResponse:
    echo_resp: Echo = call_rpc(
      self.session,
      f"{self.url}/list_relics",
      Echo(message = "ListRelicsRequest", base64_string=message_to_b64(_ListRelicsRequest), rpc_name = "list_relics")
    )
    if echo_resp is None:
      return None

    _ListRelicsResponse = ListRelicsResponse() # predefine the output proto
    _ListRelicsResponse = b64_to_message(echo_resp.base64_string, _ListRelicsResponse)
    return _ListRelicsResponse

  def update_relic_meta(self, _Relic: Relic) -> Acknowledge:
    echo_resp: Echo = call_rpc(
      self.session,
      f"{self.url}/update_relic_meta",
      Echo(message = "Relic", base64_string=message_to_b64(_Relic), rpc_name = "update_relic_meta")
    )
    if echo_resp is None:
      return None

    _Acknowledge = Acknowledge() # predefine the output proto
    _Acknowledge = b64_to_message(echo_resp.base64_string, _Acknowledge)
    return _Acknowledge

  def delete_relic(self, _Relic: Relic) -> Acknowledge:
    echo_resp: Echo = call_rpc(
      self.session,
      f"{self.url}/delete_relic",
      Echo(message = "Relic", base64_string=message_to_b64(_Relic), rpc_name = "delete_relic")
    )
    if echo_resp is None:
      return None

    _Acknowledge = Acknowledge() # predefine the output proto
    _Acknowledge = b64_to_message(echo_resp.base64_string, _Acknowledge)
    return _Acknowledge

  def get_relic_details(self, _Relic: Relic) -> Relic:
    echo_resp: Echo = call_rpc(
      self.session,
      f"{self.url}/get_relic_details",
      Echo(message = "Relic", base64_string=message_to_b64(_Relic), rpc_name = "get_relic_details")
    )
    if echo_resp is None:
      return None

    _Relic = Relic() # predefine the output proto
    _Relic = b64_to_message(echo_resp.base64_string, _Relic)
    return _Relic

  def create_file(self, _RelicFile: RelicFile) -> RelicFile:
    echo_resp: Echo = call_rpc(
      self.session,
      f"{self.url}/create_file",
      Echo(message = "RelicFile", base64_string=message_to_b64(_RelicFile), rpc_name = "create_file")
    )
    if echo_resp is None:
      return None

    _RelicFile = RelicFile() # predefine the output proto
    _RelicFile = b64_to_message(echo_resp.base64_string, _RelicFile)
    return _RelicFile

  def list_relic_files(self, _ListRelicFilesRequest: ListRelicFilesRequest) -> ListRelicFilesResponse:
    echo_resp: Echo = call_rpc(
      self.session,
      f"{self.url}/list_relic_files",
      Echo(message = "ListRelicFilesRequest", base64_string=message_to_b64(_ListRelicFilesRequest), rpc_name = "list_relic_files")
    )
    if echo_resp is None:
      return None

    _ListRelicFilesResponse = ListRelicFilesResponse() # predefine the output proto
    _ListRelicFilesResponse = b64_to_message(echo_resp.base64_string, _ListRelicFilesResponse)
    return _ListRelicFilesResponse

  def delete_multi_files(self, _RelicFiles: RelicFiles) -> Acknowledge:
    echo_resp: Echo = call_rpc(
      self.session,
      f"{self.url}/delete_multi_files",
      Echo(message = "RelicFiles", base64_string=message_to_b64(_RelicFiles), rpc_name = "delete_multi_files")
    )
    if echo_resp is None:
      return None

    _Acknowledge = Acknowledge() # predefine the output proto
    _Acknowledge = b64_to_message(echo_resp.base64_string, _Acknowledge)
    return _Acknowledge

  def download_file(self, _RelicFile: RelicFile) -> RelicFile:
    echo_resp: Echo = call_rpc(
      self.session,
      f"{self.url}/download_file",
      Echo(message = "RelicFile", base64_string=message_to_b64(_RelicFile), rpc_name = "download_file")
    )
    if echo_resp is None:
      return None

    _RelicFile = RelicFile() # predefine the output proto
    _RelicFile = b64_to_message(echo_resp.base64_string, _RelicFile)
    return _RelicFile

  def get_activity_log(self, _ActivityLogRequest: ActivityLogRequest) -> ActivityLogResponse:
    echo_resp: Echo = call_rpc(
      self.session,
      f"{self.url}/get_activity_log",
      Echo(message = "ActivityLogRequest", base64_string=message_to_b64(_ActivityLogRequest), rpc_name = "get_activity_log")
    )
    if echo_resp is None:
      return None

    _ActivityLogResponse = ActivityLogResponse() # predefine the output proto
    _ActivityLogResponse = b64_to_message(echo_resp.base64_string, _ActivityLogResponse)
    return _ActivityLogResponse

  def list_buckets(self, _ListBucketRequest: ListBucketRequest) -> ListBucketResponse:
    echo_resp: Echo = call_rpc(
      self.session,
      f"{self.url}/list_buckets",
      Echo(message = "ListBucketRequest", base64_string=message_to_b64(_ListBucketRequest), rpc_name = "list_buckets")
    )
    if echo_resp is None:
      return None

    _ListBucketResponse = ListBucketResponse() # predefine the output proto
    _ListBucketResponse = b64_to_message(echo_resp.base64_string, _ListBucketResponse)
    return _ListBucketResponse


# ------ End Stub ------ #