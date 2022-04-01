"""
gRPC + Protobuf = ♥️
"""

from grpc import RpcError
from google.protobuf.timestamp_pb2 import Timestamp
from google.protobuf.json_format import MessageToDict, MessageToJson, ParseDict

from .utils import logger

def rpc(stub: callable, message, err_msg: str, raise_on_error: bool = True):
  """convienience function for calling a gRPC stub"""
  try:
    resp = stub(message)
    raise_on_error = False
  except RpcError as e:
    logger.error(err_msg)
    # err_ = loads(e.debug_error_string())
    # if "value" in err_:
    #   if int(err_["value"]) > 500:
    #     logger.error("There is something wrong from our side. Your files are safe on your local machine.")
    #   elif int(err_["value"]) > 400:
    #     logger.error("There is something wrong in nbox. Raise an issue on github: https://github.com/NimbleBoxAI/nbox/issues")
    # logger.error(err_["grpc_message"])
    logger.error(e)
  else:
    return resp
  if raise_on_error:
    raise RpcError("NBX-RPC error, see above for details")

def streaming_rpc(stub: callable, message, err_msg: str, raise_on_error: bool = True):
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


def get_current_timestamp():
  ts = Timestamp()
  ts.GetCurrentTime()
  return ts
