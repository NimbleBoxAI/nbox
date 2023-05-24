# this will be eventually merged with the project in the root scope
from nbox.init import nbox_ws_v1
from nbox.utils import logger
from nbox.auth import secret
from nbox.relics import Relics
from nbox import messages as mpb

from nbox.lmao_v4 import get_project, get_lmao_stub
from nbox.lmao_v4.proto.lmao_service_pb2_grpc import LMAOStub
from nbox.lmao_v4.proto import tracker_pb2 as t_pb
from nbox.lmao_v4.proto import project_pb2 as p_pb
from nbox.lmao_v4.tracker import Tracker


class Project_v4:
  def __init__(self, id: str = ""):
    # id = id or ProjectState.project_id
    if not id:
      raise ValueError("Project ID is not set")
    logger.info(f"Connecting to Project: {id}")
    self.pid = id

    # self.stub = nbox_ws_v1.projects.u(id)
    # self.data = self.stub()
    self.lmao_stub = get_lmao_stub()
    _p = p_pb.Project(
      id = self.pid
    )
    self.project_pb = self.lmao_stub.GetProject(_p)
    if self.project_pb is None:
      raise ValueError("Could not connect to Monitoring backend.")
    # self.relic = Relics(id = self.project_pb.relic_id)
    self.workspace_id = secret.workspace_id

  def __repr__(self) -> str:
    return f"Project({self.pid})"

  # @property
  # def metadata(self):
  #   """A NimbleBox project is a very large entity and its components are in multiple places.
  #   `metadata` is a dictionary that contains all the information about the project."""
  #   return {"details": self.data, "lmao": mpb.MessageToDict(self.project_pb, including_default_value_fields=True)}

  # def put_settings(self, project_name: str = "", project_description: str = ""):
  #   self.stub(
  #     "put",
  #     project_name = project_name or self.data["project_name"],
  #     project_description = project_description or self.data["project_description"]
  #   )
  #   self.data["project_name"] = project_name or self.data["project_name"]
  #   self.data["project_description"] = project_description or self.data["project_description"]

  def get_lmao_stub(self) -> LMAOStub:
    return self.lmao_stub
  
  def get_tracker(self, tracker_type: int, tracker_id: str, config) -> Tracker:
    return Tracker(
      project_id = self.project_pb,
      tracker_id = tracker_id,
      config = config,
      live_tracker = tracker_type == t_pb.TrackerType.LIVE,
    )

  def get_exp_tracker(
      self,
      experiment_id: str = "",
      metadata = {},
    ) -> Tracker:
    return self.get_tracker(
      tracker_type = t_pb.TrackerType.EXPERIMENT,
      tracker_id = experiment_id,
      config = metadata
    )

  def get_live_tracker(self, serving_id: str = "", metadata = {}) -> Tracker:
    return self.get_tracker(
      tracker_type = t_pb.TrackerType.LIVE,
      tracker_id = serving_id,
      config = metadata
    )
