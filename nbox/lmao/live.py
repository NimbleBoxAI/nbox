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
from nbox.lmao.lmao_rpc_client import (
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
  """
  Get a single log by key from the serving. Logs are stored in the following format:
  `<project_id>/<serving_id>/<key>`

  Args:
    project_id: ID of the project to which the serving belongs.
    serving_id: ID of the serving to which the log belongs.
    key: Key of the log to get.
    after: Return only logs after this timestamp.
    before: Return only logs before this timestamp.
    limit: Maximum number of logs to return.

  Returns:
    The log.
  """
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
    """
    {% CallOut variant="warning" label="This is experimental and not ready for production use, can be removed without notice" /%}
    
    Client module for LMAO Live monitoring pipeline, eventually this will be merged with the LMAO client
    which is used for experiment tracking, into a single combined client.
    """
    self.workspace_id = secret(AuthConfig.workspace_id)
    self.project_id = project_id
    self.serving_id = serving_id
    
    # get the stub
    self.lmao = get_lmao_stub()
    p = ProjectProto(
      workspace_id = self.workspace_id,
      project_id = self.project_id
    )
    self.project = self.lmao.get_project(p)
    if self.project is None:
      raise Exception("Could not connect to LMAO, please check your credentials")
    
    agent_details = AgentDetails(
      workspace_id = self.workspace_id,
      nbx_serving_id = "local",
      nbx_model_id = U.SimplerTimes.get_now_str(),
    )
    self.agent = agent_details

    if serving_id:
      action = "Updating"
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
    else:
      action = "Creating"
      s: ServingProto = self.lmao.init_serving(InitRunRequest(
        workspace_id = self.workspace_id,
        project_id = self.project_id,
        config = dumps(metadata),
        agent_details = self.agent,
      ))

    logger.info(
      f"{action} live tracker\n"
      f" project: {project_id}\n"
      f"      id: {s.serving_id}\n"
      f"    link: {secret(AuthConfig.url)}/workspace/{self.workspace_id}/projects/{project_id}/#Live\n"
    )

    self.serving = s
    self._total_logged_elements  = 0 # total number of elements logged

  @property
  def serving_config(self) -> Dict[str, Any]:
    return loads(self.serving.config)


  def log(self, y: Dict[str, Union[int, float, str]]):
    run_log = RunLog(
      workspace_id = self.workspace_id,
      project_id=self.project_id,
      serving_id = self.serving.serving_id,
      log_type = RunLog.LogType.USER
    )
    for k,v in y.items():
      # TODO: @yashbonde replace Record with RecordColumn
      record = get_record(k, v)
      run_log.data.append(record)

    ack = self.lmao.on_serving_log(run_log)
    if not ack.success:
      logger.error(f"  >> Server Error\n{ack.message}")
      raise Exception("Server Error")

    self._total_logged_elements += 1


# helper class