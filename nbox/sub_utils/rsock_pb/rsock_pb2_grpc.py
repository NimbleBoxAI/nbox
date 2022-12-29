# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import nbox.sub_utils.rsock_pb.rsock_pb2 as rsock__pb2


class RSockStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.Handshake = channel.unary_unary(
                '/rsock.RSock/Handshake',
                request_serializer=rsock__pb2.HandshakeRequest.SerializeToString,
                response_deserializer=rsock__pb2.HandshakeResponse.FromString,
                )
        self.Tunnel = channel.stream_stream(
                '/rsock.RSock/Tunnel',
                request_serializer=rsock__pb2.DataPacket.SerializeToString,
                response_deserializer=rsock__pb2.DataPacket.FromString,
                )


class RSockServicer(object):
    """Missing associated documentation comment in .proto file."""

    def Handshake(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def Tunnel(self, request_iterator, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_RSockServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'Handshake': grpc.unary_unary_rpc_method_handler(
                    servicer.Handshake,
                    request_deserializer=rsock__pb2.HandshakeRequest.FromString,
                    response_serializer=rsock__pb2.HandshakeResponse.SerializeToString,
            ),
            'Tunnel': grpc.stream_stream_rpc_method_handler(
                    servicer.Tunnel,
                    request_deserializer=rsock__pb2.DataPacket.FromString,
                    response_serializer=rsock__pb2.DataPacket.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'rsock.RSock', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class RSock(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def Handshake(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/rsock.RSock/Handshake',
            rsock__pb2.HandshakeRequest.SerializeToString,
            rsock__pb2.HandshakeResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def Tunnel(request_iterator,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.stream_stream(request_iterator, target, '/rsock.RSock/Tunnel',
            rsock__pb2.DataPacket.SerializeToString,
            rsock__pb2.DataPacket.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
