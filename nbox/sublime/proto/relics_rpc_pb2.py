# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: proto/relics_rpc.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from nbox.sublime.proto import relics_pb2 as proto_dot_relics__pb2
from nbox.sublime.proto import common_pb2 as proto_dot_common__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x16proto/relics_rpc.proto\x1a\x12proto/relics.proto\x1a\x12proto/common.proto\"H\n\x12\x43reateRelicRequest\x12\x14\n\x0cworkspace_id\x18\x01 \x01(\t\x12\x0c\n\x04name\x18\x02 \x01(\t\x12\x0e\n\x06region\x18\x03 \x01(\t\"N\n\x11ListRelicsRequest\x12\x14\n\x0cworkspace_id\x18\x01 \x01(\t\x12\x12\n\nrelic_name\x18\x02 \x01(\t\x12\x0f\n\x07page_no\x18\x03 \x01(\x05\"B\n\x12ListRelicsResponse\x12\x16\n\x06relics\x18\x01 \x03(\x0b\x32\x06.Relic\x12\x14\n\x0ctotal_relics\x18\x02 \x01(\x05\"\x87\x01\n\x15ListRelicFilesRequest\x12\x14\n\x0cworkspace_id\x18\x01 \x01(\t\x12\x10\n\x08relic_id\x18\x02 \x01(\t\x12\x12\n\nrelic_name\x18\x03 \x01(\t\x12\x0e\n\x06prefix\x18\x04 \x01(\t\x12\x11\n\tfile_name\x18\x05 \x01(\t\x12\x0f\n\x07page_no\x18\x06 \x01(\x05\"H\n\x16ListRelicFilesResponse\x12\x19\n\x05\x66iles\x18\x01 \x03(\x0b\x32\n.RelicFile\x12\x13\n\x0btotal_files\x18\x02 \x01(\x05\x32\xbd\x03\n\nRelicStore\x12-\n\x0c\x63reate_relic\x12\x13.CreateRelicRequest\x1a\x06.Relic\"\x00\x12\x38\n\x0blist_relics\x12\x12.ListRelicsRequest\x1a\x13.ListRelicsResponse\"\x00\x12+\n\x11update_relic_meta\x12\x06.Relic\x1a\x0c.Acknowledge\"\x00\x12&\n\x0c\x64\x65lete_relic\x12\x06.Relic\x1a\x0c.Acknowledge\"\x00\x12%\n\x11get_relic_details\x12\x06.Relic\x1a\x06.Relic\"\x00\x12\'\n\x0b\x63reate_file\x12\n.RelicFile\x1a\n.RelicFile\"\x00\x12\x45\n\x10list_relic_files\x12\x16.ListRelicFilesRequest\x1a\x17.ListRelicFilesResponse\"\x00\x12/\n\x11\x64\x65lete_relic_file\x12\n.RelicFile\x1a\x0c.Acknowledge\"\x00\x12)\n\rdownload_file\x12\n.RelicFile\x1a\n.RelicFile\"\x00\x62\x06proto3')



_CREATERELICREQUEST = DESCRIPTOR.message_types_by_name['CreateRelicRequest']
_LISTRELICSREQUEST = DESCRIPTOR.message_types_by_name['ListRelicsRequest']
_LISTRELICSRESPONSE = DESCRIPTOR.message_types_by_name['ListRelicsResponse']
_LISTRELICFILESREQUEST = DESCRIPTOR.message_types_by_name['ListRelicFilesRequest']
_LISTRELICFILESRESPONSE = DESCRIPTOR.message_types_by_name['ListRelicFilesResponse']
CreateRelicRequest = _reflection.GeneratedProtocolMessageType('CreateRelicRequest', (_message.Message,), {
  'DESCRIPTOR' : _CREATERELICREQUEST,
  '__module__' : 'proto.relics_rpc_pb2'
  # @@protoc_insertion_point(class_scope:CreateRelicRequest)
  })
_sym_db.RegisterMessage(CreateRelicRequest)

ListRelicsRequest = _reflection.GeneratedProtocolMessageType('ListRelicsRequest', (_message.Message,), {
  'DESCRIPTOR' : _LISTRELICSREQUEST,
  '__module__' : 'proto.relics_rpc_pb2'
  # @@protoc_insertion_point(class_scope:ListRelicsRequest)
  })
_sym_db.RegisterMessage(ListRelicsRequest)

ListRelicsResponse = _reflection.GeneratedProtocolMessageType('ListRelicsResponse', (_message.Message,), {
  'DESCRIPTOR' : _LISTRELICSRESPONSE,
  '__module__' : 'proto.relics_rpc_pb2'
  # @@protoc_insertion_point(class_scope:ListRelicsResponse)
  })
_sym_db.RegisterMessage(ListRelicsResponse)

ListRelicFilesRequest = _reflection.GeneratedProtocolMessageType('ListRelicFilesRequest', (_message.Message,), {
  'DESCRIPTOR' : _LISTRELICFILESREQUEST,
  '__module__' : 'proto.relics_rpc_pb2'
  # @@protoc_insertion_point(class_scope:ListRelicFilesRequest)
  })
_sym_db.RegisterMessage(ListRelicFilesRequest)

ListRelicFilesResponse = _reflection.GeneratedProtocolMessageType('ListRelicFilesResponse', (_message.Message,), {
  'DESCRIPTOR' : _LISTRELICFILESRESPONSE,
  '__module__' : 'proto.relics_rpc_pb2'
  # @@protoc_insertion_point(class_scope:ListRelicFilesResponse)
  })
_sym_db.RegisterMessage(ListRelicFilesResponse)

_RELICSTORE = DESCRIPTOR.services_by_name['RelicStore']
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _CREATERELICREQUEST._serialized_start=66
  _CREATERELICREQUEST._serialized_end=138
  _LISTRELICSREQUEST._serialized_start=140
  _LISTRELICSREQUEST._serialized_end=218
  _LISTRELICSRESPONSE._serialized_start=220
  _LISTRELICSRESPONSE._serialized_end=286
  _LISTRELICFILESREQUEST._serialized_start=289
  _LISTRELICFILESREQUEST._serialized_end=424
  _LISTRELICFILESRESPONSE._serialized_start=426
  _LISTRELICFILESRESPONSE._serialized_end=498
  _RELICSTORE._serialized_start=501
  _RELICSTORE._serialized_end=946
# @@protoc_insertion_point(module_scope)