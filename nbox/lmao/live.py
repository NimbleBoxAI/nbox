"""
NimbleBox LMAO is our general purpose observability tool for any kind of computation you might have.
"""

# drift detection and all
# run.log_dataset(
#     dataset_name='train',
#     features=X_train,
#     predictions=y_pred_train,
#     actuals=y_train,
# )
# run.log_dataset(
#     dataset_name='test',
#     features=X_test,
#     predictions=y_pred_test,
#     actuals=y_test,
# )

from json import dumps, loads
from typing import Dict, Any, List, Optional, Union
from google.protobuf.timestamp_pb2 import Timestamp

import nbox.utils as U
from nbox.utils import logger
from nbox.auth import secret, AuthConfig

# all the sublime -> hyperloop stuff
from nbox.sublime.lmao_rpc_client import (
  InitRunRequest,
  RunLog,
  Project as ProjectProto,
  Serving as ServingProto,
  ServingLogRequest,
  ServingLogResponse,
  AgentDetails,
)
from nbox.lmao.common import get_lmao_stub, get_record

# functional parts

def get_serving_log(
  project_id: str,
  serving_id: str = "",
  key: str = "",
  after: Timestamp = None,
  before: Timestamp = None,
  limit: int = 100,
) -> ServingLogResponse:
  lmao = get_lmao_stub()
  req = ServingLogRequest(
    workspace_id = secret(AuthConfig.workspace_id),
    project_id = project_id,
    serving_id = serving_id,
    key = key,
    limit = limit,
  )
  if after:
    req.after.CopyFrom(after)
  if before:
    req.before.CopyFrom(before)
  out: ServingLogResponse = lmao.get_serving_log(req)
  return out


# the main class

class LmaoLive():
  def __init__(
    self,
    project_id: str,
    serving_id: str = "",
    metadata: Dict[str, Any] = {},
  ):
    self.workspace_id = secret(AuthConfig.workspace_id)
    self.project_id = project_id
    self.serving_id = serving_id
    
    # get the stub
    self.lmao = get_lmao_stub()
    self.project = self.lmao.get_project(ProjectProto(
      workspace_id = self.workspace_id,
      project_id = self.project_id
    ))
    if self.project is None:
      raise Exception(f"Project with id {self.project_id} does not exist")
    
    agent_details = AgentDetails(
      workspace_id = self.workspace_id,
      nbx_serving_id = "local",
      nbx_model_id = U.SimplerTimes.get_now_str(),
    )
    # if type(self.agent) == JobDetails:
    #   a: JobDetails = self.agent
    #   agent_details.type = AgentDetails.NBX.JOB
    #   agent_details.nbx_job_id = a.job_id
    #   agent_details.nbx_run_id = a.run_id
    self.agent = agent_details

    if serving_id:
      s = ServingProto(
        workspace_id = self.workspace_id,
        project_id = self.project_id,
        serving_id = self.serving_id,
        agent = self.agent,
        update_keys = ["agent",],
      )
      if metadata:
        s.config = dumps(metadata)
        s.update_keys.append("config")
      out = self.lmao.update_serving_status(s)
      if out is None:
        raise Exception(f"Failed to update serving {serving_id}, does this serving exist?")
      logger.info("Updated serving with id: " + s.serving_id)
    else:
      s: ServingProto = self.lmao.init_serving(InitRunRequest(
        workspace_id = self.workspace_id,
        project_id = self.project_id,
        config = dumps(metadata),
        agent_details = self.agent,
      ))
      logger.info("Created new serving with id: " + s.serving_id)

    self.serving = s
    self._total_logged_elements  = 0 # total number of elements logged

  def log(self, y: Dict[str, Union[int, float, str]]):
    run_log = RunLog(
      workspace_id = self.workspace_id,
      project_id=self.project_id,
      serving_id = self.serving.serving_id,
      log_type = RunLog.LogType.USER
    )
    for k,v in y.items():
      # TODO:@yashbonde replace Record with RecordColumn
      record = get_record(k, v)
      run_log.data.append(record)

    ack = self.lmao.on_serving_log(run_log)
    if not ack.success:
      logger.error(f"  >> Server Error\n{ack.message}")
      raise Exception("Server Error")

    self._total_logged_elements += 1


class __LmaoBundle:
  # files = Dict[str, Any]
  def __init__(self, name: str, meta: Dict[str, Any]):
    # bundle = LmaoBundle("bundle_name")
    # bundle = LmaoBundle("bundle_name", {"key": "value"})
    pass

  def add_files() -> None:
    # bundle.add_files("model.pt")
    # bundle.add_files("model/*.pt")
    # bundle.add_files("model/config.json", "model/tokenizer.json")
    pass

  def upload() -> None:
    # bundle.upload()
    pass

  def download() -> str:
    # bundle.download() -> "nbx_bundle_{id}" directory downloaded
    # bundle.download("model.pt") -> "nbx_bundle_{id}/model.pt"
    # bundle.download(to = "this_folder/") -> "this_folder"
    # bundle.download("model.pt", to = "this_folder/") -> "this_folder/model.pt"
    pass

  def trail() -> List[str]:
    # return a list of all the modifier of artifacts
    pass

  def verify() -> bool:
    # bundle.verify("local_folder/") -> True
    # bundle.verify("modified_local_folder/") -> False
    pass
