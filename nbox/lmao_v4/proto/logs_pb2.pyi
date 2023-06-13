"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
"""
import builtins
import google.protobuf.descriptor
import google.protobuf.internal.containers
import google.protobuf.internal.enum_type_wrapper
import google.protobuf.message
import google.protobuf.struct_pb2
import google.protobuf.timestamp_pb2
import google.protobuf.wrappers_pb2
try:
  from gen.proto import tracker_pb2 as tracker__pb2
except ImportError:
  from nbox.lmao_v4.proto import tracker_pb2
import typing
import typing_extensions

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor

class TrackerLog(google.protobuf.message.Message):
    """
    Log Messages. All the messages here are responsible for the data that is to be logged

    """
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    WORKSPACE_ID_FIELD_NUMBER: builtins.int
    PROJECT_ID_FIELD_NUMBER: builtins.int
    TRACKER_ID_FIELD_NUMBER: builtins.int
    NUMBER_KEYS_FIELD_NUMBER: builtins.int
    NUMBER_VALUES_FIELD_NUMBER: builtins.int
    TEXT_KEYS_FIELD_NUMBER: builtins.int
    TEXT_VALUES_FIELD_NUMBER: builtins.int
    TIMESTAMP_FIELD_NUMBER: builtins.int
    LOG_ID_FIELD_NUMBER: builtins.int
    workspace_id: typing.Text
    """the workspace this is part of"""

    project_id: typing.Text
    """the unique ID for this project"""

    tracker_id: typing.Text
    """the tracker ID for which you need this data"""

    @property
    def number_keys(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[typing.Text]:
        """keys for the number values"""
        pass
    @property
    def number_values(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.float]:
        """values for the number values"""
        pass
    @property
    def text_keys(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[typing.Text]:
        """keys for the text values"""
        pass
    @property
    def text_values(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[typing.Text]:
        """values for the text values"""
        pass
    @property
    def timestamp(self) -> google.protobuf.timestamp_pb2.Timestamp:
        """these two things are optional, and server can populate these
        the timestamp for this log
        """
        pass
    log_id: typing.Text
    """the UUID for this log"""

    def __init__(self,
        *,
        workspace_id: typing.Text = ...,
        project_id: typing.Text = ...,
        tracker_id: typing.Text = ...,
        number_keys: typing.Optional[typing.Iterable[typing.Text]] = ...,
        number_values: typing.Optional[typing.Iterable[builtins.float]] = ...,
        text_keys: typing.Optional[typing.Iterable[typing.Text]] = ...,
        text_values: typing.Optional[typing.Iterable[typing.Text]] = ...,
        timestamp: typing.Optional[google.protobuf.timestamp_pb2.Timestamp] = ...,
        log_id: typing.Text = ...,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["timestamp",b"timestamp"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["log_id",b"log_id","number_keys",b"number_keys","number_values",b"number_values","project_id",b"project_id","text_keys",b"text_keys","text_values",b"text_values","timestamp",b"timestamp","tracker_id",b"tracker_id","workspace_id",b"workspace_id"]) -> None: ...
global___TrackerLog = TrackerLog

class TrackerLogId(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    LOG_ID_FIELD_NUMBER: builtins.int
    log_id: typing.Text
    def __init__(self,
        *,
        log_id: typing.Text = ...,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["log_id",b"log_id"]) -> None: ...
global___TrackerLogId = TrackerLogId

class RecordColumn(google.protobuf.message.Message):
    """RecordColumn should be the main key:value(s) thing we are going to use it for both the charts and scalar data."""
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    class _DataType:
        ValueType = typing.NewType('ValueType', builtins.int)
        V: typing_extensions.TypeAlias = ValueType
    class _DataTypeEnumTypeWrapper(google.protobuf.internal.enum_type_wrapper._EnumTypeWrapper[RecordColumn._DataType.ValueType], builtins.type):
        DESCRIPTOR: google.protobuf.descriptor.EnumDescriptor
        UNSET: RecordColumn._DataType.ValueType  # 0
        NUMBER: RecordColumn._DataType.ValueType  # 1
        STRING: RecordColumn._DataType.ValueType  # 2
    class DataType(_DataType, metaclass=_DataTypeEnumTypeWrapper):
        """type of this data"""
        pass

    UNSET: RecordColumn.DataType.ValueType  # 0
    NUMBER: RecordColumn.DataType.ValueType  # 1
    STRING: RecordColumn.DataType.ValueType  # 2

    class RecordItem(google.protobuf.message.Message):
        """the actual data"""
        DESCRIPTOR: google.protobuf.descriptor.Descriptor
        X_FIELD_NUMBER: builtins.int
        TIMESTAMP_FIELD_NUMBER: builtins.int
        NUMBER_FIELD_NUMBER: builtins.int
        TEXT_FIELD_NUMBER: builtins.int
        LOG_ID_FIELD_NUMBER: builtins.int
        x: builtins.int
        """also can be called "X-axis" or just x"""

        @property
        def timestamp(self) -> google.protobuf.timestamp_pb2.Timestamp:
            """when this data was logged"""
            pass
        number: builtins.float
        """the actual number value"""

        text: typing.Text
        """the actual text value"""

        log_id: typing.Text
        """the unique tracker ID for with which data was logged"""

        def __init__(self,
            *,
            x: builtins.int = ...,
            timestamp: typing.Optional[google.protobuf.timestamp_pb2.Timestamp] = ...,
            number: builtins.float = ...,
            text: typing.Text = ...,
            log_id: typing.Text = ...,
            ) -> None: ...
        def HasField(self, field_name: typing_extensions.Literal["timestamp",b"timestamp"]) -> builtins.bool: ...
        def ClearField(self, field_name: typing_extensions.Literal["log_id",b"log_id","number",b"number","text",b"text","timestamp",b"timestamp","x",b"x"]) -> None: ...

    KEY_FIELD_NUMBER: builtins.int
    VALUE_TYPE_FIELD_NUMBER: builtins.int
    ROWS_FIELD_NUMBER: builtins.int
    key: typing.Text
    """name of this data"""

    value_type: global___RecordColumn.DataType.ValueType
    @property
    def rows(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___RecordColumn.RecordItem]:
        """give me all the values for a given column (key), so in case of table, we can simply say that
        we are returning only one value for each column ie. len(RecordColumn.rows) == 1
        this is similar to saying bar chart value, it will be a scalar but wrapped as a array of length one
        """
        pass
    def __init__(self,
        *,
        key: typing.Text = ...,
        value_type: global___RecordColumn.DataType.ValueType = ...,
        rows: typing.Optional[typing.Iterable[global___RecordColumn.RecordItem]] = ...,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["key",b"key","rows",b"rows","value_type",b"value_type"]) -> None: ...
global___RecordColumn = RecordColumn

class RecordRow(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    TRACKER_ID_FIELD_NUMBER: builtins.int
    COLUMNS_FIELD_NUMBER: builtins.int
    tracker_id: typing.Text
    @property
    def columns(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___RecordColumn]: ...
    def __init__(self,
        *,
        tracker_id: typing.Text = ...,
        columns: typing.Optional[typing.Iterable[global___RecordColumn]] = ...,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["columns",b"columns","tracker_id",b"tracker_id"]) -> None: ...
global___RecordRow = RecordRow

class TrackerLogRequest(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    WORKSPACE_ID_FIELD_NUMBER: builtins.int
    PROJECT_ID_FIELD_NUMBER: builtins.int
    TRACKER_ID_FIELD_NUMBER: builtins.int
    KEYS_FIELD_NUMBER: builtins.int
    AFTER_FIELD_NUMBER: builtins.int
    BEFORE_FIELD_NUMBER: builtins.int
    LIMIT_FIELD_NUMBER: builtins.int
    LOG_IDS_FIELD_NUMBER: builtins.int
    workspace_id: typing.Text
    """the workspace this is part of"""

    project_id: typing.Text
    """the unique ID for this project"""

    tracker_id: typing.Text
    """the tracker ID for which you need this data"""

    @property
    def keys(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[typing.Text]:
        """requested keys"""
        pass
    @property
    def after(self) -> google.protobuf.timestamp_pb2.Timestamp:
        """after this time"""
        pass
    @property
    def before(self) -> google.protobuf.timestamp_pb2.Timestamp:
        """before this time"""
        pass
    limit: builtins.int
    """limit the number of results"""

    @property
    def log_ids(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[typing.Text]:
        """the log ids that you want to get"""
        pass
    def __init__(self,
        *,
        workspace_id: typing.Text = ...,
        project_id: typing.Text = ...,
        tracker_id: typing.Text = ...,
        keys: typing.Optional[typing.Iterable[typing.Text]] = ...,
        after: typing.Optional[google.protobuf.timestamp_pb2.Timestamp] = ...,
        before: typing.Optional[google.protobuf.timestamp_pb2.Timestamp] = ...,
        limit: builtins.int = ...,
        log_ids: typing.Optional[typing.Iterable[typing.Text]] = ...,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["after",b"after","before",b"before"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["after",b"after","before",b"before","keys",b"keys","limit",b"limit","log_ids",b"log_ids","project_id",b"project_id","tracker_id",b"tracker_id","workspace_id",b"workspace_id"]) -> None: ...
global___TrackerLogRequest = TrackerLogRequest

class TrackerLogResponse(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    DATA_FIELD_NUMBER: builtins.int
    @property
    def data(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___RecordColumn]: ...
    def __init__(self,
        *,
        data: typing.Optional[typing.Iterable[global___RecordColumn]] = ...,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["data",b"data"]) -> None: ...
global___TrackerLogResponse = TrackerLogResponse

class GetTrackerTableRequest(google.protobuf.message.Message):
    """the big table object"""
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    WORKSPACE_ID_FIELD_NUMBER: builtins.int
    PROJECT_ID_FIELD_NUMBER: builtins.int
    TRACKER_TYPE_FIELD_NUMBER: builtins.int
    OFFSET_FIELD_NUMBER: builtins.int
    LIMIT_FIELD_NUMBER: builtins.int
    workspace_id: typing.Text
    project_id: typing.Text
    tracker_type: tracker_pb2.TrackerType.ValueType
    offset: builtins.int
    """offset"""

    limit: builtins.int
    """limit"""

    def __init__(self,
        *,
        workspace_id: typing.Text = ...,
        project_id: typing.Text = ...,
        tracker_type: tracker_pb2.TrackerType.ValueType = ...,
        offset: builtins.int = ...,
        limit: builtins.int = ...,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["limit",b"limit","offset",b"offset","project_id",b"project_id","tracker_type",b"tracker_type","workspace_id",b"workspace_id"]) -> None: ...
global___GetTrackerTableRequest = GetTrackerTableRequest

class TrackerTable(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    RECORD_ROWS_FIELD_NUMBER: builtins.int
    @property
    def record_rows(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___RecordRow]: ...
    def __init__(self,
        *,
        record_rows: typing.Optional[typing.Iterable[global___RecordRow]] = ...,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["record_rows",b"record_rows"]) -> None: ...
global___TrackerTable = TrackerTable

class TrackerDatasetRequest(google.protobuf.message.Message):
    """the powerfule new dataset functionality"""
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    class _ExportFormat:
        ValueType = typing.NewType('ValueType', builtins.int)
        V: typing_extensions.TypeAlias = ValueType
    class _ExportFormatEnumTypeWrapper(google.protobuf.internal.enum_type_wrapper._EnumTypeWrapper[TrackerDatasetRequest._ExportFormat.ValueType], builtins.type):
        DESCRIPTOR: google.protobuf.descriptor.EnumDescriptor
        UNSET_EF: TrackerDatasetRequest._ExportFormat.ValueType  # 0
        JSONL: TrackerDatasetRequest._ExportFormat.ValueType  # 1
        """STRUCT_PB = 1;
        CSV = 2;
        """

    class ExportFormat(_ExportFormat, metaclass=_ExportFormatEnumTypeWrapper):
        """Now our dataset system should be able to export the dataset many different ways
        currently we support JSONL but we should be able to support as many types as the
        user might prefer
        """
        pass

    UNSET_EF: TrackerDatasetRequest.ExportFormat.ValueType  # 0
    JSONL: TrackerDatasetRequest.ExportFormat.ValueType  # 1
    """STRUCT_PB = 1;
    CSV = 2;
    """


    class LogIds(google.protobuf.message.Message):
        """now we will chose any one of the following either users can give their own query or provide the log UUIDs"""
        DESCRIPTOR: google.protobuf.descriptor.Descriptor
        LOG_IDS_FIELD_NUMBER: builtins.int
        @property
        def log_ids(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[typing.Text]: ...
        def __init__(self,
            *,
            log_ids: typing.Optional[typing.Iterable[typing.Text]] = ...,
            ) -> None: ...
        def ClearField(self, field_name: typing_extensions.Literal["log_ids",b"log_ids"]) -> None: ...

    RELICS_ID_FIELD_NUMBER: builtins.int
    RELICS_PATH_FIELD_NUMBER: builtins.int
    EXPORT_FORMAT_FIELD_NUMBER: builtins.int
    LOG_REQUEST_FIELD_NUMBER: builtins.int
    QUERY_FIELD_NUMBER: builtins.int
    LOG_IDS_FIELD_NUMBER: builtins.int
    EXTRA_FIELD_NUMBER: builtins.int
    relics_id: typing.Text
    """target location details, in future this can become a more detailed message in itself
    for now we have limited the scope to store the chunks as jsonl objects that then the
    user can chose to perform things on
    """

    relics_path: typing.Text
    export_format: global___TrackerDatasetRequest.ExportFormat.ValueType
    @property
    def log_request(self) -> global___TrackerLogRequest: ...
    @property
    def query(self) -> google.protobuf.struct_pb2.Struct: ...
    @property
    def log_ids(self) -> global___TrackerDatasetRequest.LogIds: ...
    @property
    def extra(self) -> google.protobuf.struct_pb2.Struct:
        """extra things that might be used in the future, you can think this is the same as the run_kwargs for silk
        or something on those lines
        """
        pass
    def __init__(self,
        *,
        relics_id: typing.Text = ...,
        relics_path: typing.Text = ...,
        export_format: global___TrackerDatasetRequest.ExportFormat.ValueType = ...,
        log_request: typing.Optional[global___TrackerLogRequest] = ...,
        query: typing.Optional[google.protobuf.struct_pb2.Struct] = ...,
        log_ids: typing.Optional[global___TrackerDatasetRequest.LogIds] = ...,
        extra: typing.Optional[google.protobuf.struct_pb2.Struct] = ...,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["extra",b"extra","log_ids",b"log_ids","log_request",b"log_request","query",b"query","query_style",b"query_style"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["export_format",b"export_format","extra",b"extra","log_ids",b"log_ids","log_request",b"log_request","query",b"query","query_style",b"query_style","relics_id",b"relics_id","relics_path",b"relics_path"]) -> None: ...
    def WhichOneof(self, oneof_group: typing_extensions.Literal["query_style",b"query_style"]) -> typing.Optional[typing_extensions.Literal["log_request","query","log_ids"]]: ...
global___TrackerDatasetRequest = TrackerDatasetRequest

class TrackerDataset(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    DATASET_ID_FIELD_NUMBER: builtins.int
    COMPLETED_FIELD_NUMBER: builtins.int
    dataset_id: typing.Text
    @property
    def completed(self) -> google.protobuf.wrappers_pb2.BoolValue: ...
    def __init__(self,
        *,
        dataset_id: typing.Text = ...,
        completed: typing.Optional[google.protobuf.wrappers_pb2.BoolValue] = ...,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["completed",b"completed"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["completed",b"completed","dataset_id",b"dataset_id"]) -> None: ...
global___TrackerDataset = TrackerDataset