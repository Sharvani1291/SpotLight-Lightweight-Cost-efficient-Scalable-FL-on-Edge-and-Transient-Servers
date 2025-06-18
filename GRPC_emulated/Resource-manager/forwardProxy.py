import asyncio
import grpc
import logging
import logging
import time
import spotlight_pb2 
import spotlight_pb2_grpc

#Leader IP
LEADER_IP = "172.18.0.3"
LEADER_PORT= "8088"
class ForwardProxy(spotlight_pb2_grpc.ModelServiceServicer):
    def __init__(self):
        #use a single long lived channel with asyncio support
        self.channel = grpc.aio.insecure_channel(LEADER_IP + ":" + LEADER_PORT)
        self.stub = spotlight_pb2_grpc.ModelServiceStub(self.channel)
    
    #add routes here to forward data to l2,l3
    
    