import grpc
import spotlight_pb2_grpc
from spotlight_pb2 import ModelRequest, WeightsUpdate
from spotlight_pb2_grpc import ModelServiceStub
import logging
import random
import asyncio
from modelGenerator import Model
import sys
import numpy as np
import os
import uuid
# Set up logging to display information about the process
#logging.basicConfig(level=logging.INFO)

# Start of the script
#logging.info("Starting the GRPC client script.")
logging.basicConfig(filename='client.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


# Emulated client class

class EmulatedClient:
    def __init__(self, ip, port,model,mb):
        self.ip = ip
        self.port = port
        self.channel = grpc.insecure_channel(f"{self.ip}:{self.port}", options=[
            ('grpc.max_send_message_length', 500 * 1024 * 1024),  # 50 MB
            ('grpc.max_receive_message_length', 500 * 1024 * 1024)  # 50 MB
        ])
        self.model=model
        self.mb=mb
        self.clientId = str(uuid.uuid4())  # Generate a unique client ID
        logging.info(f"Emulated client initialized with ID: {self.clientId}")
    
    
    #Register the client with the proxy server
    async def register_client(self):
        stub = spotlight_pb2_grpc.ModelServiceStub(self.channel)
        try:
            response = stub.RegisterClient(client_id=self.clientId)
            logging.info(f"Client registered with ID: {self.clientId}")
            return response
        except Exception as e:
            logging.error(f"Error registering client: {e}")
            return None
    # Fetch the model from the server   

    async def fetch_model(self):
        stub = spotlight_pb2_grpc.ModelServiceStub(self.channel)
        try:
            stub.GetModel(ModelRequest())
            logging.debug(f"Received model from server using the emulated client")
            return True
        except Exception as e:
            logging.error(f"Error fetching model from server: {e}")
            return False
    
    # Send the weights emulated training by sleep and send it to the server. 
    async def send_weights(self):
        stub = spotlight_pb2_grpc.ModelServiceStub(self.channel)
        try:
            # Generate random weights for the model
            #weights=Model(self.model).generate_random_weights()
            weights=Model("cnn", dtype=np.float64, target_mb=self.mb).generate_random_weights()
            #
            #model_weights = {k: spotlight_pb2.ModelWeights(values=v.flatten().tolist()) for k, v in weights.items()}
            # for i in range(5):
            #     #random_time = random.randint(1, 2)
            #     random_sample = random.randint(50, 5000)
            #     #await asyncio.sleep(random_time)
            random_sample = random.randint(50, 5000)
            #random_time = random.randint(1, 2)
            #await asyncio.sleep(random_time)
            ack= await stub.UpdateModel(WeightsUpdate(model_weights=weights, num_samples=random_sample,model_type=self.model))
            #await asyncio.sleep(random_time)
            
            if ack.success:
                logging.info("Model update successful.")
            else:
                logging.warning(f"Model update failed: {ack.message}")

            logging.info("Client exiting.")
            return
            
            # Check if the accuracy threshold has been reached, can add this later..
        except Exception as e:
            logging.error(f"Error sending weights to server: {e}")
            
    async def run(self):
        logging.info("Starting the emulated client.")
        logging.info("Fetching model from server.")
        if await self.fetch_model():
            logging.info("Model fetched successfully.")
            await self.send_weights()

        else:
            logging.error("Error fetching model from server.")
            return
        logging.info("Emulated client process completed.")
    
