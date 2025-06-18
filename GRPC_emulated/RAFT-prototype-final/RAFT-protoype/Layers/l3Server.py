import grpc
import logging
import spotlight_pb2 
import spotlight_pb2_grpc
import logging
import uuid
import time
from datetime import datetime
from google.protobuf.empty_pb2 import Empty
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.server_api import ServerApi
import asyncio
import os
from dotenv import load_dotenv
import numpy as np

#Useraddd modules
from modelGenerator import Model
from Algorithms.fedavg import FedAvg

# Set up logging to display information about the process
logging.basicConfig(level=logging.INFO)

# Start of the script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.info("Starting the GRPC client script.")

  

#use .env file to store the connection string

class L3server(spotlight_pb2_grpc.ModelServiceServicer):
    def __init__(self):
        #loads the env variables
        load_dotenv()
        self.uri=os.getenv("MONGO_URI")
        self.db_name=os.getenv("MONGO_DB_EMULATED")
        self.collection_name=os.getenv("MONGO_COLLECTION_EMULATED")
        self.client = AsyncIOMotorClient(self.uri, maxPoolSize=20, minPoolSize=10)
        self.request_counter=0
        # Select the database and collection
        self.db = self.client[self.db_name]
        self.collection = self.db[self.collection_name]
        #self.collection.create_index("timestamp")
    
    async def fetch_latest_weights(self):
        # Connect to the MongoDB cluster
        #client = AsyncIOMotorClient(self.uri, maxPoolSize=20, minPoolSize=10)
        # Select the database and collection
        #db = client[self.db]
        #collection = db[self.collection]
        # Find the latest document in the collection
        latest_weights = await self.collection.find_one({'layer': 'L2'}, sort=[('timestamp', -1)], projection={'model_weights': 1, '_id': 0})
        latest_version = await self.collection.find_one({'layer': 'L3'}, sort=[('timestamp', -1)], projection={'version': 1, '_id': 0})
        logging.info(f"Latest version: {latest_version}")
        #logging.info(f"Latest weights: {latest_weights}")
        # Return the weights
        return latest_weights,latest_version
    
    #This function gets the latest model weights from the L2 server(Current global model)
    async def getLatestModel(self,model):
        logging.info("Received a request for the model from L2 for global aggregation.")
        try:
            latest_weights,latest_version=await self.fetch_latest_weights()
            
            if latest_weights is None:
                global_model=Model(str(model)).generate_random_weights()
            else:
                global_model = np.array(latest_weights)
                logging.info("Fetched the latest weights from the db")
            return global_model,latest_version
        except Exception as e:
            logging.error(f"Error fetching the model: {e}")
        
    async def updateL3(self, request, context):
        logging.info("Received request from L2 to aggregate the middle model.")
        try:
            # # Connect to the MongoDB cluster
            # client = AsyncIOMotorClient(self.uri)
            # # # Select the database and collection
            # db = client[self.db]
            # collection = db[self.collection]
            # # Find the latest document in the collection
            model_weights = request.model_weights
            num_samples = request.num_samples
            model=str(request.model_type)
            #calling the latest weights from L2
            logging.info("Getting the latest model weights from L2.")
            latest_weights,latest_version = await self.getLatestModel(model)
            
            if latest_version is None:
                version = 1
                logging.info(f"The current version is: {version}")
                #global_model=Model(request.model_type).generate_random_weights()
            else:
                
                version = latest_version['version'] + 1
                logging.info(f"The current version for L3 is: {version}")
                #global_model = np.array(latest_weights.get('model_weights', []))
                logging.info("Fetched the latest weights from the db")
            self.request_counter+=1
            model_weights_array = np.array(model_weights)
            start_time = time.time()
            fedavg=FedAvg(latest_weights)
                
            updated_global_model=fedavg.aggregate([model_weights_array],num_samples)
            end_time = time.time()
                
            aggregation_time=end_time-start_time
            logging.info(f"Time taken to aggregate the model weights: {aggregation_time} seconds")
            
            logging.info("Aggregated the model weights.")
            model_weights_list=updated_global_model.tolist()
                # Convert the model weights to a format suitable for MongoDB
                #version = latest_weights.get('version', 0) + 1
            document = {
                    '_id': uuid.uuid4().hex,
                    'model_weights': model_weights_list,
                    'aggregation_time': aggregation_time,
                    'version': version,
                    'number_of_samples': num_samples,
                    "layer": "L3",
                    'emulated_clients': "Yes",
                    'timestamp': datetime.now()
                }
    
                # Insert the document into the collection
            results=await self.collection.insert_one(document)
            logging.info(f"Model sent successfully. Document ID: {results.inserted_id}")
            
            
            
            return spotlight_pb2.UpdateResponse(message="Model sent successfully")
        except Exception as e:
            logging.error(f"Error sending the model: {e}")
            context.set_details(f"Error sending the model: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            return spotlight_pb2.UpdateResponse(message="Error sending the model")

# Create a gRPC server
class L3Start:
    def __init__(self):
        pass
    async def serveL3(self):
        server = grpc.aio.server( options=[
                ('grpc.max_send_message_length', 500 * 1024 * 1024),  # 50 MB
                ('grpc.max_receive_message_length', 500 * 1024 * 1024)  # 50 MB
            ])
        spotlight_pb2_grpc.add_ModelServiceServicer_to_server(L3server(), server)
        server.add_insecure_port('0.0.0.0:50052')
        await server.start()
        logging.info("Server started at port 50052 .")
        logging.info("Server started.")
        await server.wait_for_termination()          

# if __name__ == '__main__':
#     asyncio.run(server())