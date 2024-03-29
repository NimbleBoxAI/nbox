"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
"""
import abc
import google.protobuf.empty_pb2
import grpc
try:
  from gen.proto import logs_pb2 as logs__pb2
except ImportError:
  from nbox.lmao_v4.proto import logs_pb2
try:
  from gen.proto import project_pb2 as project__pb2
except ImportError:
  from nbox.lmao_v4.proto import project_pb2
try:
  from gen.proto import rules_pb2 as rules__pb2
except ImportError:
  from nbox.lmao_v4.proto import rules_pb2
try:
  from gen.proto import tracker_pb2 as tracker__pb2
except ImportError:
  from nbox.lmao_v4.proto import tracker_pb2

class LMAOStub:
    """This is the service definition for the LMAO server, this server will talk to the MongoDB in the backend."""
    def __init__(self, channel: grpc.Channel) -> None: ...
    InitProject: grpc.UnaryUnaryMultiCallable[
        project_pb2.InitProjectRequest,
        project_pb2.InitProjectResponse]
    """Project level information -> Full CRUDL on project"""

    GetProject: grpc.UnaryUnaryMultiCallable[
        project_pb2.Project,
        project_pb2.Project]

    UpdateProject: grpc.UnaryUnaryMultiCallable[
        project_pb2.Project,
        project_pb2.Project]

    DeleteProject: grpc.UnaryUnaryMultiCallable[
        project_pb2.Project,
        google.protobuf.empty_pb2.Empty]

    ListProjects: grpc.UnaryUnaryMultiCallable[
        project_pb2.ListProjectsRequest,
        project_pb2.ListProjectsResponse]

    GetRuleBuilder: grpc.UnaryUnaryMultiCallable[
        rules_pb2.RuleBuilder,
        rules_pb2.RuleBuilder]
    """Rules based workflow -> Full CRUDL for rules"""

    CreateRule: grpc.UnaryUnaryMultiCallable[
        rules_pb2.InitRuleRequest,
        rules_pb2.Rule]

    GetRule: grpc.UnaryUnaryMultiCallable[
        rules_pb2.Rule,
        rules_pb2.Rule]

    UpdateRule: grpc.UnaryUnaryMultiCallable[
        rules_pb2.Rule,
        rules_pb2.Rule]

    DeleteRule: grpc.UnaryUnaryMultiCallable[
        rules_pb2.Rule,
        google.protobuf.empty_pb2.Empty]

    ListRules: grpc.UnaryUnaryMultiCallable[
        rules_pb2.RulesList,
        rules_pb2.RulesList]

    InitTracker: grpc.UnaryUnaryMultiCallable[
        tracker_pb2.InitTrackerRequest,
        tracker_pb2.Tracker]
    """Generic Tracking -> Full CRUDL on trackers"""

    GetTracker: grpc.UnaryUnaryMultiCallable[
        tracker_pb2.Tracker,
        tracker_pb2.Tracker]

    UpdateTracker: grpc.UnaryUnaryMultiCallable[
        tracker_pb2.Tracker,
        tracker_pb2.Tracker]

    DeleteTracker: grpc.UnaryUnaryMultiCallable[
        tracker_pb2.Tracker,
        google.protobuf.empty_pb2.Empty]

    ListTrackers: grpc.UnaryUnaryMultiCallable[
        tracker_pb2.ListTrackersRequest,
        tracker_pb2.ListTrackersResponse]

    GetTrackerTable: grpc.UnaryUnaryMultiCallable[
        logs_pb2.GetTrackerTableRequest,
        logs_pb2.TrackerTable]
    """things for logs"""

    PutTrackerLog: grpc.UnaryUnaryMultiCallable[
        logs_pb2.TrackerLog,
        logs_pb2.TrackerLogId]

    GetTrackerLogs: grpc.UnaryUnaryMultiCallable[
        logs_pb2.TrackerLogRequest,
        logs_pb2.TrackerLogResponse]

    CreateDataset: grpc.UnaryUnaryMultiCallable[
        logs_pb2.TrackerDatasetRequest,
        logs_pb2.TrackerDataset]
    """
    This is the new service that we want to add

    """

    CreateDatasetStatus: grpc.UnaryUnaryMultiCallable[
        logs_pb2.TrackerDataset,
        logs_pb2.TrackerDataset]


class LMAOServicer(metaclass=abc.ABCMeta):
    """This is the service definition for the LMAO server, this server will talk to the MongoDB in the backend."""
    @abc.abstractmethod
    def InitProject(self,
        request: project_pb2.InitProjectRequest,
        context: grpc.ServicerContext,
    ) -> project_pb2.InitProjectResponse:
        """Project level information -> Full CRUDL on project"""
        pass

    @abc.abstractmethod
    def GetProject(self,
        request: project_pb2.Project,
        context: grpc.ServicerContext,
    ) -> project_pb2.Project: ...

    @abc.abstractmethod
    def UpdateProject(self,
        request: project_pb2.Project,
        context: grpc.ServicerContext,
    ) -> project_pb2.Project: ...

    @abc.abstractmethod
    def DeleteProject(self,
        request: project_pb2.Project,
        context: grpc.ServicerContext,
    ) -> google.protobuf.empty_pb2.Empty: ...

    @abc.abstractmethod
    def ListProjects(self,
        request: project_pb2.ListProjectsRequest,
        context: grpc.ServicerContext,
    ) -> project_pb2.ListProjectsResponse: ...

    @abc.abstractmethod
    def GetRuleBuilder(self,
        request: rules_pb2.RuleBuilder,
        context: grpc.ServicerContext,
    ) -> rules_pb2.RuleBuilder:
        """Rules based workflow -> Full CRUDL for rules"""
        pass

    @abc.abstractmethod
    def CreateRule(self,
        request: rules_pb2.InitRuleRequest,
        context: grpc.ServicerContext,
    ) -> rules_pb2.Rule: ...

    @abc.abstractmethod
    def GetRule(self,
        request: rules_pb2.Rule,
        context: grpc.ServicerContext,
    ) -> rules_pb2.Rule: ...

    @abc.abstractmethod
    def UpdateRule(self,
        request: rules_pb2.Rule,
        context: grpc.ServicerContext,
    ) -> rules_pb2.Rule: ...

    @abc.abstractmethod
    def DeleteRule(self,
        request: rules_pb2.Rule,
        context: grpc.ServicerContext,
    ) -> google.protobuf.empty_pb2.Empty: ...

    @abc.abstractmethod
    def ListRules(self,
        request: rules_pb2.RulesList,
        context: grpc.ServicerContext,
    ) -> rules_pb2.RulesList: ...

    @abc.abstractmethod
    def InitTracker(self,
        request: tracker_pb2.InitTrackerRequest,
        context: grpc.ServicerContext,
    ) -> tracker_pb2.Tracker:
        """Generic Tracking -> Full CRUDL on trackers"""
        pass

    @abc.abstractmethod
    def GetTracker(self,
        request: tracker_pb2.Tracker,
        context: grpc.ServicerContext,
    ) -> tracker_pb2.Tracker: ...

    @abc.abstractmethod
    def UpdateTracker(self,
        request: tracker_pb2.Tracker,
        context: grpc.ServicerContext,
    ) -> tracker_pb2.Tracker: ...

    @abc.abstractmethod
    def DeleteTracker(self,
        request: tracker_pb2.Tracker,
        context: grpc.ServicerContext,
    ) -> google.protobuf.empty_pb2.Empty: ...

    @abc.abstractmethod
    def ListTrackers(self,
        request: tracker_pb2.ListTrackersRequest,
        context: grpc.ServicerContext,
    ) -> tracker_pb2.ListTrackersResponse: ...

    @abc.abstractmethod
    def GetTrackerTable(self,
        request: logs_pb2.GetTrackerTableRequest,
        context: grpc.ServicerContext,
    ) -> logs_pb2.TrackerTable:
        """things for logs"""
        pass

    @abc.abstractmethod
    def PutTrackerLog(self,
        request: logs_pb2.TrackerLog,
        context: grpc.ServicerContext,
    ) -> logs_pb2.TrackerLogId: ...

    @abc.abstractmethod
    def GetTrackerLogs(self,
        request: logs_pb2.TrackerLogRequest,
        context: grpc.ServicerContext,
    ) -> logs_pb2.TrackerLogResponse: ...

    @abc.abstractmethod
    def CreateDataset(self,
        request: logs_pb2.TrackerDatasetRequest,
        context: grpc.ServicerContext,
    ) -> logs_pb2.TrackerDataset:
        """
        This is the new service that we want to add

        """
        pass

    @abc.abstractmethod
    def CreateDatasetStatus(self,
        request: logs_pb2.TrackerDataset,
        context: grpc.ServicerContext,
    ) -> logs_pb2.TrackerDataset: ...


def add_LMAOServicer_to_server(servicer: LMAOServicer, server: grpc.Server) -> None: ...
