"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
"""
import builtins
import google.protobuf.descriptor
import google.protobuf.internal.containers
import google.protobuf.internal.enum_type_wrapper
import google.protobuf.message
import google.protobuf.timestamp_pb2
import proto.relics_pb2
import typing
import typing_extensions

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor

class LogBuffer(google.protobuf.message.Message):
    """since this is becoming a more serious thing than what we initially were building we need to make it
    more production ready and the first step is to log a buffer object instead of the individual log object.
    we create a message called LogBuffer which is nothing but an array of _LogObject which in turn is
    either a RunLog or a LiveLog object.
    """
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    AGENT_TOKEN_FIELD_NUMBER: builtins.int
    RUN_LOGS_FIELD_NUMBER: builtins.int
    LIVE_LOGS_FIELD_NUMBER: builtins.int
    agent_token: typing.Text
    """this is like the experiment_id in runs where an agent tries to connect and server gives it a
    one time token with which all the incoming logs will be identified with
    """

    @property
    def run_logs(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___RunLog]:
        """this can either contain one or the other"""
        pass
    @property
    def live_logs(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___ServingHTTPLog]: ...
    def __init__(self,
        *,
        agent_token: typing.Text = ...,
        run_logs: typing.Optional[typing.Iterable[global___RunLog]] = ...,
        live_logs: typing.Optional[typing.Iterable[global___ServingHTTPLog]] = ...,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["agent_token",b"agent_token","live_logs",b"live_logs","run_logs",b"run_logs"]) -> None: ...
global___LogBuffer = LogBuffer

class ServingHTTPLog(google.protobuf.message.Message):
    """LiveLog is an individual call event which is created by the user agent whenever a certain API is called.
    it contains some basic information as in any other general monitoring solution.
    """
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    PATH_FIELD_NUMBER: builtins.int
    METHOD_FIELD_NUMBER: builtins.int
    STATUS_CODE_FIELD_NUMBER: builtins.int
    LATENCY_MS_FIELD_NUMBER: builtins.int
    TIMESTAMP_FIELD_NUMBER: builtins.int
    path: typing.Text
    """the endpoint ie. /api/monitoring/"""

    method: typing.Text
    """the method ie. GET, POST, PUT, DELETE"""

    status_code: builtins.int
    """the status code ie. 200, 404, 500"""

    latency_ms: builtins.float
    """the latency in milli-seconds"""

    @property
    def timestamp(self) -> google.protobuf.timestamp_pb2.Timestamp:
        """the timestamp when this event was created"""
        pass
    def __init__(self,
        *,
        path: typing.Text = ...,
        method: typing.Text = ...,
        status_code: builtins.int = ...,
        latency_ms: builtins.float = ...,
        timestamp: typing.Optional[google.protobuf.timestamp_pb2.Timestamp] = ...,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["timestamp",b"timestamp"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["latency_ms",b"latency_ms","method",b"method","path",b"path","status_code",b"status_code","timestamp",b"timestamp"]) -> None: ...
global___ServingHTTPLog = ServingHTTPLog

class ListDeploymentsRequest(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    WORKSPACE_ID_FIELD_NUMBER: builtins.int
    SERVING_ID_OR_NAME_FIELD_NUMBER: builtins.int
    PAGE_NO_FIELD_NUMBER: builtins.int
    workspace_id: typing.Text
    serving_id_or_name: typing.Text
    """used specifically for searching, if empty, it will return all the projects"""

    page_no: builtins.int
    def __init__(self,
        *,
        workspace_id: typing.Text = ...,
        serving_id_or_name: typing.Text = ...,
        page_no: builtins.int = ...,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["page_no",b"page_no","serving_id_or_name",b"serving_id_or_name","workspace_id",b"workspace_id"]) -> None: ...
global___ListDeploymentsRequest = ListDeploymentsRequest

class ListDeploymentsResponse(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    class Serving(google.protobuf.message.Message):
        DESCRIPTOR: google.protobuf.descriptor.Descriptor
        SERVING_ID_FIELD_NUMBER: builtins.int
        SERVING_NAME_FIELD_NUMBER: builtins.int
        TOTAL_MODELS_FIELD_NUMBER: builtins.int
        serving_id: typing.Text
        """the unique ID for this serving"""

        serving_name: typing.Text
        """the usermodifiable name of the serving"""

        total_models: builtins.int
        """the total number of models in this serving group"""

        def __init__(self,
            *,
            serving_id: typing.Text = ...,
            serving_name: typing.Text = ...,
            total_models: builtins.int = ...,
            ) -> None: ...
        def ClearField(self, field_name: typing_extensions.Literal["serving_id",b"serving_id","serving_name",b"serving_name","total_models",b"total_models"]) -> None: ...

    SERVING_FIELD_NUMBER: builtins.int
    TOTAL_PAGES_FIELD_NUMBER: builtins.int
    @property
    def serving(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___ListDeploymentsResponse.Serving]: ...
    total_pages: builtins.int
    def __init__(self,
        *,
        serving: typing.Optional[typing.Iterable[global___ListDeploymentsResponse.Serving]] = ...,
        total_pages: builtins.int = ...,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["serving",b"serving","total_pages",b"total_pages"]) -> None: ...
global___ListDeploymentsResponse = ListDeploymentsResponse

class ListServingsRequest(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    WORKSPACE_ID_FIELD_NUMBER: builtins.int
    DEPLOYMENT_ID_FIELD_NUMBER: builtins.int
    RUN_ID_FIELD_NUMBER: builtins.int
    PAGE_NO_FIELD_NUMBER: builtins.int
    DESC_FIELD_NUMBER: builtins.int
    workspace_id: typing.Text
    """the workspace this is part of"""

    deployment_id: typing.Text
    """the unique ID for this project"""

    run_id: typing.Text
    """for searching"""

    page_no: builtins.int
    desc: builtins.bool
    def __init__(self,
        *,
        workspace_id: typing.Text = ...,
        deployment_id: typing.Text = ...,
        run_id: typing.Text = ...,
        page_no: builtins.int = ...,
        desc: builtins.bool = ...,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["deployment_id",b"deployment_id","desc",b"desc","page_no",b"page_no","run_id",b"run_id","workspace_id",b"workspace_id"]) -> None: ...
global___ListServingsRequest = ListServingsRequest

class ListServingsResponse(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    MODEL_IDS_FIELD_NUMBER: builtins.int
    KEYS_FIELD_NUMBER: builtins.int
    CREATED_AT_FIELD_NUMBER: builtins.int
    TOTAL_PAGES_FIELD_NUMBER: builtins.int
    @property
    def model_ids(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[typing.Text]:
        """the list of runs in this project"""
        pass
    @property
    def keys(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[typing.Text]:
        """the list of chart-keys for this run"""
        pass
    @property
    def created_at(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.int]:
        """time when these runs were created"""
        pass
    total_pages: builtins.int
    """the total number of pages"""

    def __init__(self,
        *,
        model_ids: typing.Optional[typing.Iterable[typing.Text]] = ...,
        keys: typing.Optional[typing.Iterable[typing.Text]] = ...,
        created_at: typing.Optional[typing.Iterable[builtins.int]] = ...,
        total_pages: builtins.int = ...,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["created_at",b"created_at","keys",b"keys","model_ids",b"model_ids","total_pages",b"total_pages"]) -> None: ...
global___ListServingsResponse = ListServingsResponse

class Serving(google.protobuf.message.Message):
    """the Big Serving Object"""
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    class _Status:
        ValueType = typing.NewType('ValueType', builtins.int)
        V: typing_extensions.TypeAlias = ValueType
    class _StatusEnumTypeWrapper(google.protobuf.internal.enum_type_wrapper._EnumTypeWrapper[Serving._Status.ValueType], builtins.type):
        DESCRIPTOR: google.protobuf.descriptor.EnumDescriptor
        NOT_SET: Serving._Status.ValueType  # 0
        """When created and added to DB but no further action taken"""

        RUNNING: Serving._Status.ValueType  # 1
        """When the first on_serving_log has been called"""

        COMPLETED: Serving._Status.ValueType  # 2
        """When there is a graceful exit on_serving_end has been called"""

    class Status(_Status, metaclass=_StatusEnumTypeWrapper):
        pass

    NOT_SET: Serving.Status.ValueType  # 0
    """When created and added to DB but no further action taken"""

    RUNNING: Serving.Status.ValueType  # 1
    """When the first on_serving_log has been called"""

    COMPLETED: Serving.Status.ValueType  # 2
    """When there is a graceful exit on_serving_end has been called"""


    AGENT_FIELD_NUMBER: builtins.int
    AGENT_TOKEN_FIELD_NUMBER: builtins.int
    CREATED_AT_FIELD_NUMBER: builtins.int
    CONFIG_FIELD_NUMBER: builtins.int
    STATUS_FIELD_NUMBER: builtins.int
    UPDATED_AT_FIELD_NUMBER: builtins.int
    @property
    def agent(self) -> global___AgentDetails:
        """the agent details"""
        pass
    agent_token: typing.Text
    """the unique ID of this agent from the MongoDB backend"""

    created_at: builtins.int
    """same as InitRunRequest.created_at"""

    config: typing.Text
    """the jsonified config string"""

    status: global___Serving.Status.ValueType
    """the last known status of this run"""

    updated_at: builtins.int
    """the last time this"""

    def __init__(self,
        *,
        agent: typing.Optional[global___AgentDetails] = ...,
        agent_token: typing.Text = ...,
        created_at: builtins.int = ...,
        config: typing.Text = ...,
        status: global___Serving.Status.ValueType = ...,
        updated_at: builtins.int = ...,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["agent",b"agent"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["agent",b"agent","agent_token",b"agent_token","config",b"config","created_at",b"created_at","status",b"status","updated_at",b"updated_at"]) -> None: ...
global___Serving = Serving

class Record(google.protobuf.message.Message):
    """This is the data that the user will be logging if this"""
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    class _DataType:
        ValueType = typing.NewType('ValueType', builtins.int)
        V: typing_extensions.TypeAlias = ValueType
    class _DataTypeEnumTypeWrapper(google.protobuf.internal.enum_type_wrapper._EnumTypeWrapper[Record._DataType.ValueType], builtins.type):
        DESCRIPTOR: google.protobuf.descriptor.EnumDescriptor
        FLOAT: Record._DataType.ValueType  # 0
        INTEGER: Record._DataType.ValueType  # 1
        STRING: Record._DataType.ValueType  # 2
    class DataType(_DataType, metaclass=_DataTypeEnumTypeWrapper):
        """since it is not possible to have the map in the message, we need to have an enum for the target datatype"""
        pass

    FLOAT: Record.DataType.ValueType  # 0
    INTEGER: Record.DataType.ValueType  # 1
    STRING: Record.DataType.ValueType  # 2

    KEY_FIELD_NUMBER: builtins.int
    VALUE_TYPE_FIELD_NUMBER: builtins.int
    STEP_FIELD_NUMBER: builtins.int
    FLOAT_DATA_FIELD_NUMBER: builtins.int
    INTEGER_DATA_FIELD_NUMBER: builtins.int
    STRING_DATA_FIELD_NUMBER: builtins.int
    key: typing.Text
    """name of this data"""

    value_type: global___Record.DataType.ValueType
    step: builtins.int
    """also can be called "X-axis" or just x"""

    @property
    def float_data(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.float]:
        """all the data is done in the"""
        pass
    @property
    def integer_data(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.int]: ...
    @property
    def string_data(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[typing.Text]: ...
    def __init__(self,
        *,
        key: typing.Text = ...,
        value_type: global___Record.DataType.ValueType = ...,
        step: builtins.int = ...,
        float_data: typing.Optional[typing.Iterable[builtins.float]] = ...,
        integer_data: typing.Optional[typing.Iterable[builtins.int]] = ...,
        string_data: typing.Optional[typing.Iterable[typing.Text]] = ...,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["float_data",b"float_data","integer_data",b"integer_data","key",b"key","step",b"step","string_data",b"string_data","value_type",b"value_type"]) -> None: ...
global___Record = Record

class RecordColumn(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    class _DataType:
        ValueType = typing.NewType('ValueType', builtins.int)
        V: typing_extensions.TypeAlias = ValueType
    class _DataTypeEnumTypeWrapper(google.protobuf.internal.enum_type_wrapper._EnumTypeWrapper[RecordColumn._DataType.ValueType], builtins.type):
        DESCRIPTOR: google.protobuf.descriptor.EnumDescriptor
        FLOAT: RecordColumn._DataType.ValueType  # 0
        INTEGER: RecordColumn._DataType.ValueType  # 1
        STRING: RecordColumn._DataType.ValueType  # 2
    class DataType(_DataType, metaclass=_DataTypeEnumTypeWrapper):
        """type of this data"""
        pass

    FLOAT: RecordColumn.DataType.ValueType  # 0
    INTEGER: RecordColumn.DataType.ValueType  # 1
    STRING: RecordColumn.DataType.ValueType  # 2

    class RecordRow(google.protobuf.message.Message):
        """the actual data"""
        DESCRIPTOR: google.protobuf.descriptor.Descriptor
        X_FIELD_NUMBER: builtins.int
        FLOAT_DATA_FIELD_NUMBER: builtins.int
        INTEGER_DATA_FIELD_NUMBER: builtins.int
        STRING_DATA_FIELD_NUMBER: builtins.int
        x: builtins.int
        """also can be called "X-axis" or just x"""

        float_data: builtins.float
        """all the data is done in the"""

        integer_data: builtins.int
        string_data: typing.Text
        def __init__(self,
            *,
            x: builtins.int = ...,
            float_data: builtins.float = ...,
            integer_data: builtins.int = ...,
            string_data: typing.Text = ...,
            ) -> None: ...
        def ClearField(self, field_name: typing_extensions.Literal["float_data",b"float_data","integer_data",b"integer_data","string_data",b"string_data","x",b"x"]) -> None: ...

    KEY_FIELD_NUMBER: builtins.int
    VALUE_TYPE_FIELD_NUMBER: builtins.int
    ROWS_FIELD_NUMBER: builtins.int
    key: typing.Text
    """since it is not possible to have the map in the message, we need to have an enum for the target datatype
    name of this data
    """

    value_type: global___RecordColumn.DataType.ValueType
    @property
    def rows(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___RecordColumn.RecordRow]: ...
    def __init__(self,
        *,
        key: typing.Text = ...,
        value_type: global___RecordColumn.DataType.ValueType = ...,
        rows: typing.Optional[typing.Iterable[global___RecordColumn.RecordRow]] = ...,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["key",b"key","rows",b"rows","value_type",b"value_type"]) -> None: ...
global___RecordColumn = RecordColumn

class RunLog(google.protobuf.message.Message):
    """This is the aggregation of all the logs for this run"""
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    class _LogType:
        ValueType = typing.NewType('ValueType', builtins.int)
        V: typing_extensions.TypeAlias = ValueType
    class _LogTypeEnumTypeWrapper(google.protobuf.internal.enum_type_wrapper._EnumTypeWrapper[RunLog._LogType.ValueType], builtins.type):
        DESCRIPTOR: google.protobuf.descriptor.EnumDescriptor
        SYSTEM: RunLog._LogType.ValueType  # 0
        """these are the mtrics like GPU/RAM utilisation"""

        NBX: RunLog._LogType.ValueType  # 1
        """these are the metrics that NimbleBox generates for the user"""

        USER: RunLog._LogType.ValueType  # 2
        """these are the metrics that the user has logged"""

    class LogType(_LogType, metaclass=_LogTypeEnumTypeWrapper):
        pass

    SYSTEM: RunLog.LogType.ValueType  # 0
    """these are the mtrics like GPU/RAM utilisation"""

    NBX: RunLog.LogType.ValueType  # 1
    """these are the metrics that NimbleBox generates for the user"""

    USER: RunLog.LogType.ValueType  # 2
    """these are the metrics that the user has logged"""


    EXPERIMENT_ID_FIELD_NUMBER: builtins.int
    DATA_FIELD_NUMBER: builtins.int
    COLUMN_DATA_FIELD_NUMBER: builtins.int
    LOG_TYPE_FIELD_NUMBER: builtins.int
    experiment_id: typing.Text
    @property
    def data(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___Record]: ...
    @property
    def column_data(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___RecordColumn]:
        """this is the chart wise data"""
        pass
    log_type: global___RunLog.LogType.ValueType
    """type of data"""

    def __init__(self,
        *,
        experiment_id: typing.Text = ...,
        data: typing.Optional[typing.Iterable[global___Record]] = ...,
        column_data: typing.Optional[typing.Iterable[global___RecordColumn]] = ...,
        log_type: global___RunLog.LogType.ValueType = ...,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["column_data",b"column_data","data",b"data","experiment_id",b"experiment_id","log_type",b"log_type"]) -> None: ...
global___RunLog = RunLog

class AgentDetails(google.protobuf.message.Message):
    """all the NBX-Infra details for this specific run"""
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    class _NBX:
        ValueType = typing.NewType('ValueType', builtins.int)
        V: typing_extensions.TypeAlias = ValueType
    class _NBXEnumTypeWrapper(google.protobuf.internal.enum_type_wrapper._EnumTypeWrapper[AgentDetails._NBX.ValueType], builtins.type):
        DESCRIPTOR: google.protobuf.descriptor.EnumDescriptor
        JOB: AgentDetails._NBX.ValueType  # 0
        SERVING: AgentDetails._NBX.ValueType  # 1
    class NBX(_NBX, metaclass=_NBXEnumTypeWrapper):
        pass

    JOB: AgentDetails.NBX.ValueType  # 0
    SERVING: AgentDetails.NBX.ValueType  # 1

    TYPE_FIELD_NUMBER: builtins.int
    NBX_JOB_ID_FIELD_NUMBER: builtins.int
    NBX_SERVING_ID_FIELD_NUMBER: builtins.int
    NBX_RUN_ID_FIELD_NUMBER: builtins.int
    NBX_MODEL_ID_FIELD_NUMBER: builtins.int
    WORKSPACE_ID_FIELD_NUMBER: builtins.int
    type: global___AgentDetails.NBX.ValueType
    """this is the type of the agent, JOB or SERVING"""

    nbx_job_id: typing.Text
    """NBX-Jobs ID"""

    nbx_serving_id: typing.Text
    """deployment id"""

    nbx_run_id: typing.Text
    """JobRun"""

    nbx_model_id: typing.Text
    """model id"""

    workspace_id: typing.Text
    def __init__(self,
        *,
        type: global___AgentDetails.NBX.ValueType = ...,
        nbx_job_id: typing.Text = ...,
        nbx_serving_id: typing.Text = ...,
        nbx_run_id: typing.Text = ...,
        nbx_model_id: typing.Text = ...,
        workspace_id: typing.Text = ...,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["id",b"id","instance",b"instance","nbx_job_id",b"nbx_job_id","nbx_model_id",b"nbx_model_id","nbx_run_id",b"nbx_run_id","nbx_serving_id",b"nbx_serving_id"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["id",b"id","instance",b"instance","nbx_job_id",b"nbx_job_id","nbx_model_id",b"nbx_model_id","nbx_run_id",b"nbx_run_id","nbx_serving_id",b"nbx_serving_id","type",b"type","workspace_id",b"workspace_id"]) -> None: ...
    @typing.overload
    def WhichOneof(self, oneof_group: typing_extensions.Literal["id",b"id"]) -> typing.Optional[typing_extensions.Literal["nbx_job_id","nbx_serving_id"]]: ...
    @typing.overload
    def WhichOneof(self, oneof_group: typing_extensions.Literal["instance",b"instance"]) -> typing.Optional[typing_extensions.Literal["nbx_run_id","nbx_model_id"]]: ...
global___AgentDetails = AgentDetails

class InitRunRequest(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    AGENT_DETAILS_FIELD_NUMBER: builtins.int
    CREATED_AT_FIELD_NUMBER: builtins.int
    CONFIG_FIELD_NUMBER: builtins.int
    PROJECT_ID_FIELD_NUMBER: builtins.int
    PROJECT_NAME_FIELD_NUMBER: builtins.int
    @property
    def agent_details(self) -> global___AgentDetails: ...
    created_at: builtins.int
    """the unix timestamp when this run was created"""

    config: typing.Text
    """the jsonified singular config string"""

    project_id: typing.Text
    """when this is an experiment we will have project details"""

    project_name: typing.Text
    def __init__(self,
        *,
        agent_details: typing.Optional[global___AgentDetails] = ...,
        created_at: builtins.int = ...,
        config: typing.Text = ...,
        project_id: typing.Text = ...,
        project_name: typing.Text = ...,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["agent_details",b"agent_details"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["agent_details",b"agent_details","config",b"config","created_at",b"created_at","project_id",b"project_id","project_name",b"project_name"]) -> None: ...
global___InitRunRequest = InitRunRequest

class File(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    NAME_FIELD_NUMBER: builtins.int
    CREATED_AT_FIELD_NUMBER: builtins.int
    IS_INPUT_FIELD_NUMBER: builtins.int
    RELIC_FILE_FIELD_NUMBER: builtins.int
    name: typing.Text
    """the relative (to **job root**) filename of the File, final location is save_location/name"""

    created_at: builtins.int
    """when was this made"""

    is_input: builtins.bool
    """this was there when the run was created or this was created by the run as an output"""

    @property
    def relic_file(self) -> proto.relics_pb2.RelicFile:
        """this is the file when used with"""
        pass
    def __init__(self,
        *,
        name: typing.Text = ...,
        created_at: builtins.int = ...,
        is_input: builtins.bool = ...,
        relic_file: typing.Optional[proto.relics_pb2.RelicFile] = ...,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["relic_file",b"relic_file"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["created_at",b"created_at","is_input",b"is_input","name",b"name","relic_file",b"relic_file"]) -> None: ...
global___File = File

class FileList(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    EXPERIMENT_ID_FIELD_NUMBER: builtins.int
    FILES_FIELD_NUMBER: builtins.int
    experiment_id: typing.Text
    """associated run"""

    @property
    def files(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___File]:
        """all the Files"""
        pass
    def __init__(self,
        *,
        experiment_id: typing.Text = ...,
        files: typing.Optional[typing.Iterable[global___File]] = ...,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["experiment_id",b"experiment_id","files",b"files"]) -> None: ...
global___FileList = FileList

class Run(google.protobuf.message.Message):
    """the Big Object"""
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    class _Status:
        ValueType = typing.NewType('ValueType', builtins.int)
        V: typing_extensions.TypeAlias = ValueType
    class _StatusEnumTypeWrapper(google.protobuf.internal.enum_type_wrapper._EnumTypeWrapper[Run._Status.ValueType], builtins.type):
        DESCRIPTOR: google.protobuf.descriptor.EnumDescriptor
        NOT_SET: Run._Status.ValueType  # 0
        """When created and added to DB but no further action taken"""

        RUNNING: Run._Status.ValueType  # 1
        """When the first on_log has been called"""

        COMPLETED: Run._Status.ValueType  # 2
        """When the on_train_end has been called"""

        FAILED: Run._Status.ValueType  # 3
        """When the NBX-Job failed"""

    class Status(_Status, metaclass=_StatusEnumTypeWrapper):
        pass

    NOT_SET: Run.Status.ValueType  # 0
    """When created and added to DB but no further action taken"""

    RUNNING: Run.Status.ValueType  # 1
    """When the first on_log has been called"""

    COMPLETED: Run.Status.ValueType  # 2
    """When the on_train_end has been called"""

    FAILED: Run.Status.ValueType  # 3
    """When the NBX-Job failed"""


    AGENT_FIELD_NUMBER: builtins.int
    EXPERIMENT_ID_FIELD_NUMBER: builtins.int
    CREATED_AT_FIELD_NUMBER: builtins.int
    ENDED_AT_FIELD_NUMBER: builtins.int
    COMPLETED_FIELD_NUMBER: builtins.int
    SAVE_LOCATION_FIELD_NUMBER: builtins.int
    FILE_LIST_FIELD_NUMBER: builtins.int
    CONFIG_FIELD_NUMBER: builtins.int
    STATUS_FIELD_NUMBER: builtins.int
    UPDATED_AT_FIELD_NUMBER: builtins.int
    @property
    def agent(self) -> global___AgentDetails:
        """the agent details"""
        pass
    experiment_id: typing.Text
    """the unique ID of this run from the MongoDB backend"""

    created_at: builtins.int
    """same as InitRunRequest.created_at"""

    ended_at: builtins.int
    """when was run declared dead, the actual kill can be well before that"""

    completed: builtins.bool
    """is this run complete"""

    save_location: typing.Text
    """this is the location where the Files from this run are stored"""

    @property
    def file_list(self) -> global___FileList:
        """all the Files from this run"""
        pass
    config: typing.Text
    """the jsonified config string"""

    status: global___Run.Status.ValueType
    """the last known status of this run"""

    updated_at: builtins.int
    """the last time this run was updated"""

    def __init__(self,
        *,
        agent: typing.Optional[global___AgentDetails] = ...,
        experiment_id: typing.Text = ...,
        created_at: builtins.int = ...,
        ended_at: builtins.int = ...,
        completed: builtins.bool = ...,
        save_location: typing.Text = ...,
        file_list: typing.Optional[global___FileList] = ...,
        config: typing.Text = ...,
        status: global___Run.Status.ValueType = ...,
        updated_at: builtins.int = ...,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["agent",b"agent","file_list",b"file_list"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["agent",b"agent","completed",b"completed","config",b"config","created_at",b"created_at","ended_at",b"ended_at","experiment_id",b"experiment_id","file_list",b"file_list","save_location",b"save_location","status",b"status","updated_at",b"updated_at"]) -> None: ...
global___Run = Run

class ListProjectsRequest(google.protobuf.message.Message):
    """
    Now we have the data structures for the user-client to get the data from the Konark.

    Request Reponse kind of thing
    $app/monitoring/
    """
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    WORKSPACE_ID_FIELD_NUMBER: builtins.int
    PROJECT_ID_OR_NAME_FIELD_NUMBER: builtins.int
    PAGE_NO_FIELD_NUMBER: builtins.int
    workspace_id: typing.Text
    """the workspace this is part of"""

    project_id_or_name: typing.Text
    """used specifically for searching, if empty, it will return all the projects"""

    page_no: builtins.int
    def __init__(self,
        *,
        workspace_id: typing.Text = ...,
        project_id_or_name: typing.Text = ...,
        page_no: builtins.int = ...,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["page_no",b"page_no","project_id_or_name",b"project_id_or_name","workspace_id",b"workspace_id"]) -> None: ...
global___ListProjectsRequest = ListProjectsRequest

class ListProjectsResponse(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    class Project(google.protobuf.message.Message):
        DESCRIPTOR: google.protobuf.descriptor.Descriptor
        PROJECT_ID_FIELD_NUMBER: builtins.int
        PROJECT_NAME_FIELD_NUMBER: builtins.int
        TOTAL_EXPERIMENTS_FIELD_NUMBER: builtins.int
        project_id: typing.Text
        """the unique ID for this project"""

        project_name: typing.Text
        """the usermodifiable name of the project"""

        total_experiments: builtins.int
        """the total number of experiments in this project"""

        def __init__(self,
            *,
            project_id: typing.Text = ...,
            project_name: typing.Text = ...,
            total_experiments: builtins.int = ...,
            ) -> None: ...
        def ClearField(self, field_name: typing_extensions.Literal["project_id",b"project_id","project_name",b"project_name","total_experiments",b"total_experiments"]) -> None: ...

    PROJECTS_FIELD_NUMBER: builtins.int
    TOTAL_PAGES_FIELD_NUMBER: builtins.int
    @property
    def projects(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___ListProjectsResponse.Project]:
        """the list of projects in this workspace"""
        pass
    total_pages: builtins.int
    """the total number of pages"""

    def __init__(self,
        *,
        projects: typing.Optional[typing.Iterable[global___ListProjectsResponse.Project]] = ...,
        total_pages: builtins.int = ...,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["projects",b"projects","total_pages",b"total_pages"]) -> None: ...
global___ListProjectsResponse = ListProjectsResponse

class ListRunsRequest(google.protobuf.message.Message):
    """$app/monitoring/gjj9dk30"""
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    WORKSPACE_ID_FIELD_NUMBER: builtins.int
    PROJECT_ID_FIELD_NUMBER: builtins.int
    EXPERIMENT_ID_FIELD_NUMBER: builtins.int
    PAGE_NO_FIELD_NUMBER: builtins.int
    DESC_FIELD_NUMBER: builtins.int
    workspace_id: typing.Text
    """the workspace this is part of"""

    project_id: typing.Text
    """the unique ID for this project"""

    experiment_id: typing.Text
    """for searching"""

    page_no: builtins.int
    desc: builtins.bool
    def __init__(self,
        *,
        workspace_id: typing.Text = ...,
        project_id: typing.Text = ...,
        experiment_id: typing.Text = ...,
        page_no: builtins.int = ...,
        desc: builtins.bool = ...,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["desc",b"desc","experiment_id",b"experiment_id","page_no",b"page_no","project_id",b"project_id","workspace_id",b"workspace_id"]) -> None: ...
global___ListRunsRequest = ListRunsRequest

class ListRunsResponse(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    EXPERIMENT_IDS_FIELD_NUMBER: builtins.int
    KEYS_FIELD_NUMBER: builtins.int
    CREATED_AT_FIELD_NUMBER: builtins.int
    TOTAL_PAGES_FIELD_NUMBER: builtins.int
    @property
    def experiment_ids(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[typing.Text]:
        """the list of runs in this project"""
        pass
    @property
    def keys(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[typing.Text]:
        """the list of chart-keys for this run"""
        pass
    @property
    def created_at(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.int]:
        """time when these runs were created"""
        pass
    total_pages: builtins.int
    """the total number of pages"""

    def __init__(self,
        *,
        experiment_ids: typing.Optional[typing.Iterable[typing.Text]] = ...,
        keys: typing.Optional[typing.Iterable[typing.Text]] = ...,
        created_at: typing.Optional[typing.Iterable[builtins.int]] = ...,
        total_pages: builtins.int = ...,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["created_at",b"created_at","experiment_ids",b"experiment_ids","keys",b"keys","total_pages",b"total_pages"]) -> None: ...
global___ListRunsResponse = ListRunsResponse

class RunLogRequest(google.protobuf.message.Message):
    """logs for $app/monitoring/gjj9dk30/experiment/cmk03kt03/"""
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    EXPERIMENT_ID_FIELD_NUMBER: builtins.int
    KEY_FIELD_NUMBER: builtins.int
    SAMPLE_FIELD_NUMBER: builtins.int
    START_AT_FIELD_NUMBER: builtins.int
    END_AT_FIELD_NUMBER: builtins.int
    experiment_id: typing.Text
    """the unique ID for this run"""

    key: typing.Text
    """the key to search for, if empty, it will return all the logs"""

    sample: builtins.int
    """the number of items to sample, default is 1500"""

    start_at: builtins.int
    """the start at, default is 0"""

    end_at: builtins.int
    """the end at, default is -1"""

    def __init__(self,
        *,
        experiment_id: typing.Text = ...,
        key: typing.Text = ...,
        sample: builtins.int = ...,
        start_at: builtins.int = ...,
        end_at: builtins.int = ...,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["end_at",b"end_at","experiment_id",b"experiment_id","key",b"key","sample",b"sample","start_at",b"start_at"]) -> None: ...
global___RunLogRequest = RunLogRequest
