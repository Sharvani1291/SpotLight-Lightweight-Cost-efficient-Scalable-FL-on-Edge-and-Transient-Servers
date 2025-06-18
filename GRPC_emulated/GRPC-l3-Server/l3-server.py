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
from bson.binary import Binary
from pymongo.server_api import ServerApi
import asyncio
import os
from dotenv import load_dotenv
import numpy as np

#Useraddd modules
from modelGenerator import Model
#from Algorithms.NumbaFedBuff import NumbaFedBuff
from Algorithms.FedProx import FedProx



# Start of the script
logging.basicConfig(filename='L3.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("Starting the GRPC client script.")

  

#use .env file to store the connection string

class L3server(spotlight_pb2_grpc.ModelServiceServicer):
    def __init__(self):
        #loads the env variables
        load_dotenv()
        self.uri=os.getenv("MONGO_URI")
        self.db=os.getenv("MONGO_DB")
        self.collection=os.getenv("MONGO_COLLECTION")
        self.client = AsyncIOMotorClient(self.uri, maxPoolSize=20, minPoolSize=10)
        self.request_counter=0
        # Select the database and collection
        self.db = self.client[self.db]
        self.collection = self.db[self.collection]
        self.updated_global_model = None
        self.aggregation_time = None
        self.aggregation_lock = asyncio.Lock()
        self.collection.create_index("timestamp")
    
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
                global_model= Model(os.getenv("model", "cnn"), dtype=np.float64,target_mb=10).generate_random_weights()
            else:
                global_model = np.array(latest_weights)
                logging.info("Fetched the latest weights from the db")
            return global_model,latest_version
        except Exception as e:
            logging.error(f"Error fetching the model: {e}")
        
    async def updateL3(self, request, context):
        logging.info("Received request from L2 to aggregate the middle model.")
        try:
            model_weights = request.model_weights
            num_samples = request.num_samples
            model = str(request.model_type)

            logging.info("Getting the latest model weights from L2.")
            global_model, latest_version = await self.getLatestModel(model)

            version = 1 if latest_version is None else latest_version['version'] + 1
            logging.info(f"The current version for L3 is: {version}")

            # Prepare input for aggregation
            model_weights_array = np.array(model_weights, dtype=np.float32)

            # Accumulate updates
            if not hasattr(self, "model_list"):
                self.model_list = []
                self.sample_counts = []

            self.model_list.append(model_weights_array)
            self.sample_counts.append(num_samples)
            self.request_counter += 1

            # Only aggregate when enough updates have arrived
            if self.request_counter >= 1:
                async with self.aggregation_lock:
                    start_time = time.time()
                    fedBuff=FedProx(global_model,mu=0.5)
                    self.updated_global_model=fedBuff.aggregate(self.model_list,self.sample_counts)
                    end_time = time.time()
                    self.aggregation_time = end_time - start_time

                    logging.info(f"Time taken to aggregate the model weights: {self.aggregation_time:.4f} seconds")
                    logging.info("Aggregated the model weights.")

                    #self.updated_global_model.tolist()
                    weights_f32 = self.updated_global_model.astype(np.float32, copy=False)
                    logging.info(f"Updated global model type: {weights_f32.dtype}")
                        #self.model_weights_list=weights_f32.tolist()
                        
                        
                        # Convert the model weights to a format suitable for MongoDB
                    blob = Binary(weights_f32.tobytes())

                    # Prepare document for MongoDB
                    document = {
                        '_id': uuid.uuid4().hex,
                        'model_weights': blob,
                        'aggregation_time': self.aggregation_time,
                        'version': version,
                        'number_of_samples': sum(self.sample_counts),
                        "layer": "L3",
                        'emulated_clients': "Yes",
                        'timestamp': datetime.now()
                    }

                    await self.collection.insert_one(document)
                    logging.info(f"Model sent successfully. Document ID: {document['_id']}")

                    # Reset for next round
                    self.model_list.clear()
                    self.sample_counts.clear()
                    self.request_counter = 0

                    return spotlight_pb2.UpdateResponse(message="Model sent successfully")
            else:
                logging.info(f"Received {self.request_counter}/5 updates. Waiting for more.")
                return spotlight_pb2.UpdateResponse(message="Waiting for more L2 updates")

        except Exception as e:
            logging.error(f"Error sending the model: {e}")
            context.set_details(f"Error sending the model: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            return spotlight_pb2.UpdateResponse(message="Error sending the model")


# Create a gRPC server
async def server():
    try:
        server = grpc.aio.server( options=[
                ('grpc.max_send_message_length', 500 * 1024 * 1024),  # 50 MB
                ('grpc.max_receive_message_length', 500 * 1024 * 1024)  # 50 MB
            ])
        spotlight_pb2_grpc.add_ModelServiceServicer_to_server(L3server(), server)
        server.add_insecure_port('[::]:50052')
        await server.start()
        logging.info("Server started at port 50052 .")
        logging.info("Server started.")
        await server.wait_for_termination()
    except Exception as e:
        logging.error(f"Error starting the server: {e}")
        raise          

if __name__ == '__main__':
    asyncio.run(server())