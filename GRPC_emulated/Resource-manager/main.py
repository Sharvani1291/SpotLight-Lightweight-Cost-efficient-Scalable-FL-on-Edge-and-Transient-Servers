import grpc
import logging
import spotlight_pb2
import spotlight_pb2_grpc
import asyncio
from dotenv import load_dotenv
from google.protobuf.empty_pb2 import Empty
import zmq
import os

# Start of the script
logging.basicConfig(filename='RM.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', mode='w')
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.info("Starting the GRPC client script.")

class ResourceManager(spotlight_pb2_grpc.ModelServiceServicer):
    def __init__(self):
        load_dotenv()
        
        #for zmq
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect("tcp://localhost:5555")
        # Initialize the connection to the L2 server
        # self.l2_uri = os.getenv("l2_uri")
        # self.channel = grpc.aio.insecure_channel(self.l2_uri)
        
        # self.l3_uri = os.getenv("l3_uri")
        # self.channel3 = grpc.aio.insecure_channel(self.l3_uri)
        
        # Routing table
        self.dns_routing_table = []
        self.current_index = 0
    
    # This constructs the routes that will be used to send the data from client to server using zmq and pub-sub
    def construct_routes(self):
    
        self.socket.send_string("Requesting routing table")
        message = self.socket.recv_string()
        
        addresses = message.split(",")
        self.dns_routing_table = addresses
        self.current_index = 0
        logging.info(f"Received the following routing table: {self.dns_routing_table}")
        return message

    def delete_routes(self):
        self.socket.send_string("Requesting routing table")
        message = self.socket.recv_string()
        
        address_to_remove = message.strip()
        if address_to_remove in self.dns_routing_table:
            self.dns_routing_table.remove(address_to_remove)
            logging.info(f"Removed address: {address_to_remove}")
        else:
            logging.warning(f"Address to remove not found in routing table: {address_to_remove}")
    # This method selects the next address in the routing table using round-robin
    def get_next_address(self):
        if not self.dns_routing_table:
            raise Exception("DNS routing table is empty.")
        address = self.dns_routing_table[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.dns_routing_table)
        return address
    
    def refresh_routes_if_needed(self):
        if not self.dns_routing_table:
            logging.info("DNS routing table is empty, refreshing routes.")
            self.construct_routes()
        else:
            logging.info("Using cached DNS routing table.")
            
            
    async def GetModel(self, request, context):
        # Then create the stub and call the GET model function and return the model its output to the client
        logging.info("Received a request for the model.")
        try:
            # Construct routes before getting the address
            self.refresh_routes_if_needed()
            
            #delete the route if the pod is deleted by the emulator
            self.delete_routes()
            #gets the next address from the routing table
            address = self.get_next_address()
            
            channel = grpc.aio.insecure_channel(address)
            stub = spotlight_pb2_grpc.ModelServiceStub(channel)
            response = await stub.GetModel(request)
            return response
        except Exception as e:
            logging.error(f"Error fetching the latest weights: {e}")
            raise

    async def UpdateModel(self, request, context):
        # Then create the stub and call the UpdateModel function and return the model its output to the client
        logging.info("Received a request to update the model at L2.")
        try:
            # Rebuild the routes if needed
            self.refresh_routes_if_needed()
            
            #delete the route if the pod is deleted by the emulator
            self.delete_routes()
            # Get the address from the get_next_address function
            address = self.get_next_address()
            
            channel = grpc.aio.insecure_channel(address)
            stub = spotlight_pb2_grpc.ModelServiceStub(channel)
            response = await stub.UpdateModel(request)
            return response
        except Exception as e:
            logging.error(f"Error fetching the latest weights: {e}")
            raise

    async def updateL3(self, request, context):
        # Then create the stub and call the UpdateModel function and return the model its output to the client
        logging.info("Received a request to send the model to L3 Server.")
        try:
            stub = spotlight_pb2_grpc.ModelServiceStub(self.channel3)
            response = await stub.updateL3(request)
            return response
        except Exception as e:
            logging.error(f"Error fetching the latest weights: {e}")
            raise       

async def serve():
    server = grpc.aio.server()
    spotlight_pb2_grpc.add_ModelServiceServicer_to_server(ResourceManager(), server)
    server.add_insecure_port('[::]:8081')
    logging.info("Server starting on port 8081")
    await server.start()
    await server.wait_for_termination()

if __name__ == '__main__':
    logging.info("Initializing the Resource Manager server.")
    asyncio.run(serve())