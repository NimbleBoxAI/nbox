import sys
import grpc
import datetime
import tabulate
from functools import lru_cache
from typing import Tuple,  Dict, Any

from google.protobuf.field_mask_pb2 import FieldMask

from nbox import messages as mpb
from nbox.utils import logger, lo
from nbox.auth import secret, auth_info_pb
from nbox.init import nbox_ws_v1, nbox_model_service_stub, nbox_serving_service_stub

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
from nbox.hyperloop.common.common_pb2 import Resource


################################################################################
"""
# NimbleBox.ai Serving

This is the proposed interface for the NimbleBox.ai Serving API. We want to keep
the highest levels of consistency with the NBX-Jobs API.
"""
################################################################################

@lru_cache()
def _get_deployment_data(name: str = "", id: str = "", *, workspace_id: str = "") -> Tuple[str, str]:
  """
  Get the deployment data, either by name or id

  Args:
    name (str, optional): Name of the deployment. Defaults to "".
    id (str, optional): ID of the deployment. Defaults to "".

  Returns:
    Tuple[str, str]: (id, name)
  """
  # print("Getting deployment data", name, id, workspace_id)
  if (not name and not id) or (name and id):
    logger.warning("Must provide either name or id")
    return None, None
  # filter and get "id" and "name"
  workspace_id = workspace_id or secret.workspace_id

  # get the deployment
  serving: Serving = mpb.rpc(
    nbox_serving_service_stub.GetServing,
    ServingRequest(
      serving=Serving(name=name, id=id),
      auth_info = auth_info_pb()
    ),
    "Could not get deployment",
    raise_on_error=True
  )

  return serving.id, serving.name

def print_serving_list(sort: str = "created_on", *, workspace_id: str = ""):
  """
  Print the list of deployments

  Args:
    sort (str, optional): Sort by. Defaults to "created_on".
  """
  def _get_time(t):
    return datetime.fromtimestamp(int(float(t))).strftime("%Y-%m-%d %H:%M:%S")

  workspace_id = workspace_id or secret.workspace_id
  all_deployments: ServingListResponse = mpb.rpc(
    nbox_serving_service_stub.ListServings,
    ServingListRequest(
      auth_info=auth_info_pb(),
      limit=10
    ),
    "Could not get deployments",
    raise_on_error=True
  )
  # sorted_depls = sorted(all_deployments, key = lambda x: x[sort], reverse = sort == "created_on")
  # headers = ["created_on", "id", "name", "pinned_id", "pinned_name", "pinned_last_updated"]
  # [TODO] add sort by create time
  headers = ["id", "name", "pinned_id", "pinned_name", "pinned_last_updated"]
  all_depls = []
  for depl in all_deployments.servings:
    _depl = (depl.id, depl.name)
    pinned = depl.models[0] if len(depl.models) else None
    if not pinned:
      _depl += (None, None,)
    else:
      _depl += (pinned.id, pinned.name, _get_time(pinned.created_at.seconds))
    all_depls.append(_depl)

  for l in tabulate.tabulate(all_depls, headers).splitlines():
    logger.info(l)


class Serve:
  # status = staticmethod(print_serving_list)
  # upload: 'Serve' = staticmethod(partial(upload_job_folder, "serving"))

  def __init__(self, serving_id: str = "", model_id: str = "", *, workspace_id: str = "") -> None:
    """Python wrapper for NBX-Serving gRPC API

    Args:
      serving_id (str, optional): Serving ID. Defaults to None.
      model_id (str, optional): Model ID. Defaults to None.
    """
    self.id = serving_id
    self.model_id = model_id
    self.workspace_id = workspace_id or secret.workspace_id
    if workspace_id is None:
      raise DeprecationWarning("Personal workspace does not support serving")
    else:
      serving_id, serving_name = _get_deployment_data(name = "", id = self.id, workspace_id = self.workspace_id) # TODO add name support
    self.serving_id = serving_id
    self.serving_name = serving_name
    self.ws_stub = nbox_ws_v1.deployments

  def __repr__(self) -> str:
    x = f"nbox.Serve('{self.id}', '{self.workspace_id}'"
    if self.model_id is not None:
      x += f", model_id = '{self.model_id}'"
    x += ")"
    return x

  # there are things that are on the serving group level
  
  def logs(self, f = sys.stdout):
    """Get the logs of the model deployment

    Args:
      f (file, optional): File to write the logs to. Defaults to sys.stdout.
    """
    logger.debug(f"Streaming logs of model '{self.model_id}'")
    for model_log in mpb.streaming_rpc(
      nbox_model_service_stub.ModelLogs,
      ModelRequest(
        model = ModelProto(
          id = self.model_id,
          serving_group_id = self.serving_id
        ),
        auth_info = auth_info_pb(),
      ),
      f"Could not get logs of model {self.model_id}, only live logs are available",
      False
    ):
      for log in model_log.log:
        f.write(log)
        f.flush()

  def change_serving_resources(self, resource: Dict[str, Any]):
    if type(resource, Resource):
      pass
    elif type(resource, dict):
      resource = mpb.dict_to_message(resource, Resource())
    else:
      raise TypeError(f"resource must be of type dict or Resource, not {type(resource)}")

    logger.info(f"Updating resources of a serving group: {self.serving_id}")
    mpb.rpc(
      nbox_serving_service_stub.UpdateServing,
      UpdateServingRequest(
        model=Serving(
          id=self.model_id,
          resource=resource
        ),
        update_mask=FieldMask(paths=["resource"]),
        auth_info = auth_info_pb()
      ),
      "Could not change resources of serving group",
      raise_on_error = True
    )
    pass

  # there are things on the model level

  def deploy(self, tag: str = "", feature_gates: Dict[str, str] = {}, resource: Dict[str, Any] = {}):
    if not self.model_id:
      raise ValueError("Model ID is required")

    if type(resource) ==  Resource:
      pass
    elif type(resource) ==  dict:
      resource = mpb.dict_to_message(resource, Resource())
    else:
      raise TypeError(f"resource must be of type dict or Resource, not {type(resource)}")

    model = ModelProto(
      id = self.model_id,
      serving_group_id = self.serving_id,
      resource = resource
    )
    if tag:
      model.feature_gates.update({"SetModelMetadata": tag})
    if feature_gates:
      model.feature_gates.update(feature_gates)
    logger.info(f"Deploying model {self.model_id} to deployment {self.serving_id} with tag: '{tag}' and feature gates: {feature_gates}")
    try:
      nbox_model_service_stub.Deploy(ModelRequest(model = model, auth_info = auth_info_pb()))
    except grpc.RpcError as e:
      logger.error(lo(
        f"Could not deploy model {self.model_id} to deployment {self.serving_id}\n",
        f"gRPC Code: {e.code()}\n"
        f"    Error: {e.details()}",
      ))

  def pin(self) -> bool:
    """Pin a model to the deployment

    Args:
      model_id (str, optional): Model ID. Defaults to None.
      workspace_id (str, optional): Workspace ID. Defaults to "".
    """
    if not self.model_id:
      raise ValueError("Model ID is required")

    logger.info(f"Pin model {self.model_id} to deployment {self.serving_id}")
    mpb.rpc(
      nbox_model_service_stub.SetModelPin,
      ModelRequest(
        model = ModelProto(
          id = self.model_id,
          serving_group_id = self.serving_id,
          pin_status = ModelProto.PinStatus.PIN_STATUS_PINNED
        ),
        auth_info = auth_info_pb()
      ),
      "Could not pin model",
      raise_on_error = True
    )
  
  def unpin(self) -> bool:
    """Pin a model to the deployment

    Args:
      model_id (str, optional): Model ID. Defaults to None.
      workspace_id (str, optional): Workspace ID. Defaults to "".
    """
    if not self.model_id:
      raise ValueError("Model ID is required")

    logger.info(f"Unpin model {self.model_id} to deployment {self.serving_id}")
    mpb.rpc(
      nbox_model_service_stub.SetModelPin,
      ModelRequest(
        model = ModelProto(
          id = self.model_id,
          serving_group_id = self.serving_id,
          pin_status = ModelProto.PinStatus.PIN_STATUS_UNPINNED
        ),
        auth_info = auth_info_pb(),
      ),
      "Could not unpin model",
      raise_on_error = True
    )
  
  def scale(self, replicas: int) -> bool:
    """Scale the model deployment

    Args:
      replicas (int): Number of replicas
    """
    if not self.model_id:
      raise ValueError("Model ID is required to scale a model. Pass with --model_id")
    if replicas < 0:
      raise ValueError("Replicas must be greater than or equal to 0")

    logger.info(f"Scale model deployment {self.model_id} to {replicas} replicas")
    mpb.rpc(
      nbox_model_service_stub.UpdateModel,
      UpdateModelRequest(
        model=ModelProto(
          id=self.model_id,
          serving_group_id=self.serving_id,
          replicas=replicas
        ),
        update_mask=FieldMask(paths=["replicas"]),
        auth_info = auth_info_pb()
      ),
      "Could not scale deployment",
      raise_on_error = True
    )

  def change_model_resources(self, resource: Dict[str, Any]):
    if type(resource) ==  Resource:
      pass
    elif type(resource) ==  dict:
      resource = mpb.dict_to_message(resource, Resource())
    else:
      raise TypeError(f"resource must be of type dict or Resource, not {type(resource)}")

    logger.warn(f"Updating resources of a running model '{self.model_id}', this will result in a restart!")
    mpb.rpc(
      nbox_model_service_stub.UpdateModel,
      UpdateModelRequest(
        model=ModelProto(
          id=self.model_id,
          serving_group_id=self.serving_id,
          resource=resource
        ),
        update_mask=FieldMask(paths=["resource"]),
        auth_info = auth_info_pb()
      ),
      "Could not change resources of a running model",
      raise_on_error = True
    )