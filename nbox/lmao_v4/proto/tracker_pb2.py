# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: tracker.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.protobuf import struct_pb2 as google_dot_protobuf_dot_struct__pb2
from google.protobuf import timestamp_pb2 as google_dot_protobuf_dot_timestamp__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\rtracker.proto\x12\x07lmao_pb\x1a\x1cgoogle/protobuf/struct.proto\x1a\x1fgoogle/protobuf/timestamp.proto\"\xde\x01\n\x12InitTrackerRequest\x12\x14\n\x0cworkspace_id\x18\x01 \x01(\t\x12\x12\n\nproject_id\x18\x02 \x01(\t\x12\x1a\n\x04type\x18\x03 \x01(\x0e\x32\x0c.lmao_pb.NBX\x12\x14\n\x0cnbx_group_id\x18\x04 \x01(\t\x12\x17\n\x0fnbx_instance_id\x18\x05 \x01(\t\x12\'\n\x06\x63onfig\x18\x06 \x01(\x0b\x32\x17.google.protobuf.Struct\x12*\n\x0ctracker_type\x18\x07 \x01(\x0e\x32\x14.lmao_pb.TrackerType\"\xd8\x03\n\x07Tracker\x12\x14\n\x0cworkspace_id\x18\x01 \x01(\t\x12\x12\n\nproject_id\x18\x02 \x01(\t\x12\n\n\x02id\x18\x03 \x01(\t\x12\'\n\x06\x63onfig\x18\x05 \x01(\x0b\x32\x17.google.protobuf.Struct\x12.\n\ncreated_at\x18\x06 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12.\n\nupdated_at\x18\x07 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12\x15\n\rsave_location\x18\t \x01(\t\x12\'\n\x06status\x18\n \x01(\x0e\x32\x17.lmao_pb.Tracker.Status\x12*\n\x0ctracker_type\x18\x0b \x01(\x0e\x32\x14.lmao_pb.TrackerType\x12\x1a\n\x04type\x18\x0c \x01(\x0e\x32\x0c.lmao_pb.NBX\x12\x14\n\x0cnbx_group_id\x18\r \x01(\t\x12\x17\n\x0fnbx_instance_id\x18\x0e \x01(\t\x12\x13\n\x0bupdate_keys\x18\x0f \x03(\t\"B\n\x06Status\x12\x10\n\x0cUNSET_STATUS\x10\x00\x12\x0b\n\x07RUNNING\x10\x01\x12\r\n\tCOMPLETED\x10\x02\x12\n\n\x06\x46\x41ILED\x10\x03\"\x81\x03\n\x13ListTrackersRequest\x12\x14\n\x0cworkspace_id\x18\x01 \x01(\t\x12\x12\n\nproject_id\x18\x02 \x01(\t\x12\x0e\n\x06offset\x18\x03 \x01(\x03\x12\r\n\x05limit\x18\x04 \x01(\x03\x12\x31\n\rcreated_after\x18\x05 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12\x32\n\x0e\x63reated_before\x18\x06 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12\x31\n\rupdated_after\x18\x07 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12\x32\n\x0eupdated_before\x18\x08 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12\'\n\x06status\x18\t \x01(\x0e\x32\x17.lmao_pb.Tracker.Status\x12*\n\x0ctracker_type\x18\n \x01(\x0e\x32\x14.lmao_pb.TrackerType\"O\n\x14ListTrackersResponse\x12\"\n\x08trackers\x18\x01 \x03(\x0b\x32\x10.lmao_pb.Tracker\x12\x13\n\x0btotal_pages\x18\x02 \x01(\x05*&\n\x03NBX\x12\t\n\x05LOCAL\x10\x00\x12\x07\n\x03JOB\x10\x01\x12\x0b\n\x07SERVING\x10\x02*7\n\x0bTrackerType\x12\x0e\n\nUNSET_TYPE\x10\x00\x12\x0e\n\nEXPERIMENT\x10\x01\x12\x08\n\x04LIVE\x10\x02\x42/Z-github.com/NimbleBoxAI/nimblebox-lmao/lmao_pbb\x06proto3')

_NBX = DESCRIPTOR.enum_types_by_name['NBX']
NBX = enum_type_wrapper.EnumTypeWrapper(_NBX)
_TRACKERTYPE = DESCRIPTOR.enum_types_by_name['TrackerType']
TrackerType = enum_type_wrapper.EnumTypeWrapper(_TRACKERTYPE)
LOCAL = 0
JOB = 1
SERVING = 2
UNSET_TYPE = 0
EXPERIMENT = 1
LIVE = 2


_INITTRACKERREQUEST = DESCRIPTOR.message_types_by_name['InitTrackerRequest']
_TRACKER = DESCRIPTOR.message_types_by_name['Tracker']
_LISTTRACKERSREQUEST = DESCRIPTOR.message_types_by_name['ListTrackersRequest']
_LISTTRACKERSRESPONSE = DESCRIPTOR.message_types_by_name['ListTrackersResponse']
_TRACKER_STATUS = _TRACKER.enum_types_by_name['Status']
InitTrackerRequest = _reflection.GeneratedProtocolMessageType('InitTrackerRequest', (_message.Message,), {
  'DESCRIPTOR' : _INITTRACKERREQUEST,
  '__module__' : 'tracker_pb2'
  # @@protoc_insertion_point(class_scope:lmao_pb.InitTrackerRequest)
  })
_sym_db.RegisterMessage(InitTrackerRequest)

Tracker = _reflection.GeneratedProtocolMessageType('Tracker', (_message.Message,), {
  'DESCRIPTOR' : _TRACKER,
  '__module__' : 'tracker_pb2'
  # @@protoc_insertion_point(class_scope:lmao_pb.Tracker)
  })
_sym_db.RegisterMessage(Tracker)

ListTrackersRequest = _reflection.GeneratedProtocolMessageType('ListTrackersRequest', (_message.Message,), {
  'DESCRIPTOR' : _LISTTRACKERSREQUEST,
  '__module__' : 'tracker_pb2'
  # @@protoc_insertion_point(class_scope:lmao_pb.ListTrackersRequest)
  })
_sym_db.RegisterMessage(ListTrackersRequest)

ListTrackersResponse = _reflection.GeneratedProtocolMessageType('ListTrackersResponse', (_message.Message,), {
  'DESCRIPTOR' : _LISTTRACKERSRESPONSE,
  '__module__' : 'tracker_pb2'
  # @@protoc_insertion_point(class_scope:lmao_pb.ListTrackersResponse)
  })
_sym_db.RegisterMessage(ListTrackersResponse)

if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'Z-github.com/NimbleBoxAI/nimblebox-lmao/lmao_pb'
  _NBX._serialized_start=1258
  _NBX._serialized_end=1296
  _TRACKERTYPE._serialized_start=1298
  _TRACKERTYPE._serialized_end=1353
  _INITTRACKERREQUEST._serialized_start=90
  _INITTRACKERREQUEST._serialized_end=312
  _TRACKER._serialized_start=315
  _TRACKER._serialized_end=787
  _TRACKER_STATUS._serialized_start=721
  _TRACKER_STATUS._serialized_end=787
  _LISTTRACKERSREQUEST._serialized_start=790
  _LISTTRACKERSREQUEST._serialized_end=1175
  _LISTTRACKERSRESPONSE._serialized_start=1177
  _LISTTRACKERSRESPONSE._serialized_end=1256
# @@protoc_insertion_point(module_scope)
