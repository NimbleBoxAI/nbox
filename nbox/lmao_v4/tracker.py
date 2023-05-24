import os
import grpc
from typing import Union, Dict, List, Any

from nbox import utils as U
from nbox.auth import secret, NBX_JOB_TYPE, NBX_DEPLOY_TYPE
from nbox.utils import logger, lo
from nbox.relics import Relics

from google.protobuf.struct_pb2 import Struct

from nbox.lmao_v4.proto.project_pb2 import Project as ProjectProto
from nbox.lmao_v4.proto.tracker_pb2 import Tracker as TrackerProto, TrackerType, InitTrackerRequest, NBX as NBXType
from nbox.lmao_v4.proto.logs_pb2 import TrackerLogId
from nbox.lmao_v4.common import get_lmao_stub, get_tracker_log


class Tracker():
  """This is the generic class for any kind of Tracker, users can submodule this and leverage the existing methods"""
  def __init__(
      self,
      project_id: Union[str, ProjectProto],
      tracker_id: Union[str, TrackerProto] = "",
      live_tracker: bool = False,
      config: Dict[str, Any] = {},
      *,
      buffer_size: int = 10
    ):
    self.stub = get_lmao_stub()
    self.project_pb = project_id
    if type(project_id) != ProjectProto:
      self.project_pb = self.stub.GetProject(ProjectProto(id = project_id))
    
    # The big tracker switch case
    if tracker_id == "":
      logger.info("Initialising NBX-tracker")
      action = "Initialised"
      self.tracker_pb = self._init_tracker(
        project_id = self.project_pb.id,
        tracker_type = TrackerType.LIVE if live_tracker else TrackerType.EXPERIMENT,
        config = config
      )
    elif type(tracker_id) == TrackerProto:
      logger.info("Loaded NBX-tracker")
      action = "Loaded"
      self.tracker_pb = tracker_id
    elif isinstance(tracker_id, str):
      logger.info("Getting NBX-tracker")
      action = "Got"
      self.tracker_pb = self.stub.GetTracker(TrackerProto(project_id = self.project_pb.id, id=tracker_id))
    else:
      raise ValueError("Invalid tracker_id")
    
    logger.info(lo(
      f"{action} NBX-tracker",
      project = self.project_pb.id,
      tracker = self.tracker_pb.id,
      link = f"{secret.nbx_url}/workspace/{self.tracker_pb.workspace_id}/projects/{self.project_pb.id}"
    ))

    self.relic = Relics(id = self.project_pb.relic_id, prefix = self.tracker_pb.save_location)
    self.total_logged_elements = 0
    self.buffer_size = buffer_size
    self.pending_log_buffer = [] # this is for the rolling buffer when 

  def __repr__(self) -> str:
    return (f"    project: {self.project_pb.id}\n"
            f" tracker-id: {self.tracker_pb.id}\n"
            f"       link: {secret.nbx_url}/workspace/{self.tracker_pb.workspace_id}/projects/{self.project_pb.id}")

  def get_relic(self):
    """Get the underlying Relic for more advanced usage patterns."""
    return self.relic

  # before even thinking we should directly expose all the underlying rpc methods
  def _init_tracker(self, project_id, tracker_type: int, config: Dict[str, Any] = {}):
    # get the agent details
    agent_details = secret.get_agent_details()
    agent_type = NBXType.LOCAL
    if agent_details == NBX_JOB_TYPE:
      agent_type = NBXType.JOB
    elif agent_details == NBX_DEPLOY_TYPE:
      agent_type = NBXType.SERVING

    # get the struct for the config
    config_struct = Struct()
    config_struct.update(config)

    # initialise the tracker and return
    tracker: TrackerProto = self.stub.InitTracker(
      InitTrackerRequest(
        project_id = project_id,
        tracker_type = tracker_type,
        nbx_group_id = agent_details.group_id,
        nbx_instance_id = agent_details.instance_id,
        type = agent_type,
        config = config_struct
      )
    )
    return tracker
  
  def update_tracker_agent(self):
    agent_details = secret.get_agent_details()
    agent_type = NBXType.LOCAL
    if agent_details.nbx_type == NBX_JOB_TYPE:
      agent_type = NBXType.JOB
    elif agent_details.nbx_type == NBX_DEPLOY_TYPE:
      agent_type = NBXType.SERVING

    logger.info(f"Updating tracker agent: {agent_details.group_id} - {agent_details.instance_id}")
    self.tracker_pb.nbx_group_id = agent_details.group_id
    self.tracker_pb.nbx_instance_id = agent_details.instance_id
    self.tracker_pb.type = agent_type
    self.tracker_pb.update_keys.extend(("nbx_group_id", "nbx_instance_id", "type"))
    self.tracker_pb: TrackerProto = self.stub.UpdateTracker(self.tracker_pb)

  def log(self, log: Dict[str, Union[int, float, str]], *, log_id: str = "") -> str:
    if self.tracker_pb.status == TrackerProto.Status.COMPLETED:
      raise ValueError("Cannot log to a completed tracker")
    tracker_log = get_tracker_log(log)
    if log_id == "":
      U.is_valid_uuid(log_id)
      tracker_log.log_id = log_id
    tracker_log.project_id = self.tracker_pb.project_id
    tracker_log.tracker_id = self.tracker_pb.id
    self.pending_log_buffer.append(log)

    try:
      for log in self.pending_log_buffer:
        log_id: TrackerLogId = self.stub.PutTrackerLog(tracker_log)
        self.total_logged_elements += 1
    except grpc.RpcError as e:
      logger.warning(f"Failed to log: {log}")
      self.pending_log_buffer.append(log)
      return ""

    return log_id.log_id

  def end(self):
    """End the tracker to declare it complete. This will then trigger all the post end operations"""
    if self.tracker_pb.status == TrackerProto.Status.COMPLETED:
      raise ValueError("Cannot end a completed tracker")
    logger.info(f"Ending project ('{self.tracker_pb.project_id}') tracker: '{self.tracker_pb.id}'")
    self.tracker_pb.status = TrackerProto.Status.COMPLETED
    self.tracker_pb.update_keys.append("status")
    self.tracker_pb = self.stub.UpdateTracker(self.tracker_pb)

  def save_file(self, *files: List[str]):
    """
    Args:
      files: The list of files to save. This can be a list of files or a list of folders. If a folder is passed, all the files in the folder will be uploaded.
    """
    logger.info(f"Saving files: {files}")
    # manage all the complexity of getting the list of RelicFile
    all_files = []
    for folder_or_file in files:
      if os.path.isfile(folder_or_file):
        all_files.append(folder_or_file)
      elif os.path.isdir(folder_or_file):
        all_files.extend(U.get_files_in_folder(folder_or_file, abs_path=False))
      else:
        raise Exception(f"File or Folder not found: {folder_or_file}")

    # when we add something cool we can use this
    logger.debug(f"Storing {len(all_files)} files")
    relic = self.get_relic()
    for f in all_files:
      relic.put(f)
    return all_files

  def add_files(self, *files: List[str]):
    logger.warning("Tracker.add_files is deprecated, please use save_file instead")
    return self.save_file(*files)
  
  def delete_tracker(self):
    logger.info(f"Deleting tracker: {self.tracker_pb.id}")
    self.stub.DeleteTracker(self.tracker_pb)
