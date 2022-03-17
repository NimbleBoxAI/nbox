from grpc import RpcError
from google.protobuf.json_format import MessageToDict, MessageToJson, ParseDict

from .utils import logger

def rpc(stub: callable, message, err_msg: str, raise_on_error: bool = False):
  """convienience function for calling a gRPC stub"""
  try:
    resp = stub(message)
  except RpcError as e:
    logger.error(err_msg)
    logger.error(e.details())
    if raise_on_error:
      raise e
  else:
    return resp

def streaming_rpc(stub: callable, message, err_msg: str, raise_on_error: bool = False):
  """convienience function for streaming from a gRPC stub"""
  try:
    data_iter = stub(message)
    for data in data_iter:
      yield data
  except RpcError as e:
    logger.error(err_msg)
    logger.error(e.details())
    if raise_on_error:
      raise e

def message_to_json(message):
  """convert message to json"""
  return MessageToJson(
    message=message,
    including_default_value_fields=True,
    preserving_proto_field_name=True,
    indent=2,
    sort_keys=False,
    use_integers_for_enums=True,
    float_precision=4
  )

def message_to_dict(message):
  """convert message to dict"""
  return MessageToDict(
    message = message,
    including_default_value_fields=True,
    preserving_proto_field_name=True,
    use_integers_for_enums=True,
    float_precision=4
  )

def dict_to_message(dict, message):
  """load dict into message"""
  ParseDict(dict, message)
  return message
