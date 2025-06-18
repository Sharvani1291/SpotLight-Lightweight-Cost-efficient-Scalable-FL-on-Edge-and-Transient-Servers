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
import asyncio
import os
from dotenv import load_dotenv
import numpy as np

#Useraddd modules
from modelGenerator import Model


# Set up logging to display information about the process
logging.basicConfig(level=logging.INFO)


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Start of the script
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.info("Starting the GRPC client script.")
#use .env file to store the connection string

class L2server(spotlight_pb2_grpc.ModelServiceServicer):
    def __init__(self):
        #loads the env variables
        load_dotenv()
        self.uri=os.getenv("MONGO_URI")
        self.db_name=os.getenv("MONGO_DB_EMULATED")
        self.collection_name=os.getenv("MONGO_COLLECTION_EMULATED")
        self.client = AsyncIOMotorClient(self.uri, maxPoolSize=20, minPoolSize=10)
        if self.client:
            logging.info("Connection to MongoDB successful.")
        # Select the database and collection
        self.db = self.client[self.db_name]
        self.collection = self.db[self.collection_name]
        self.model_list=[]
        #self.collection.create_index("timestamp")

    async def startup_tasks(self):
        """ Ensure index creation happens asynchronously """
        await self.collection.create_index("timestamp", name="timestamp_index", background=True)
        logging.info("Index on 'timestamp' created successfully.")

    
    #This function fetches the latest weights from the database and inserts the new weights,I made one function to create just one pool
    async def fetch_latest_weights(self,function_term,payload=None):
        try:
        # Connect to the MongoDB cluster
            if function_term=="latest":
            # Find the latest document in the collection
                latest_weights = await self.collection.find_one(sort=[('timestamp', -1)])
                logging.info("Fetched the latest weights from the db")
                # Return the weights
                return latest_weights
            if function_term=="insert":
                result=await self.collection.insert_one(payload)
                if result.acknowledged:
                    logging.info(f"Model sent inserted successfully. Document ID: {result.inserted_id}")
                if result is None or not result.acknowledged:
                    logging.error("Failed to insert the document into MongoDB.")
                #logging.info(f"Model sent successfully. Document ID: {result.inserted_id}")
        except Exception as e:
            logging.error(f"Error fetching the latest weights: {e}")
            raise 
    async def GetModel(self, request, context):
        logging.info("Received a request for the model.")
        try:
            await self.fetch_latest_weights("latest")
            return Empty()
        except Exception as e:
            logging.error(f"Error fetching the latest weights: {e}")
            context.set_details(f"Error fetching the latest weights: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            return Empty()
        
    async def UpdateModel(self, request, context):
        logging.info("Received a request to send the model.")
        try:
            model_weights = request.model_weights
            num_samples = request.num_samples
            #model=str(request.model_type)
            logging.info(f"Received the model type: {request.model_type}")
            logging.debug(f"Received the model with the type: {type(request.model_type)}")
            #calling the latest weights
            latest_weights = await self.fetch_latest_weights("latest")
            if latest_weights is None:
                version = 1
                logging.info(f"The current version is: {version}")
                global_model=Model(request.model_type).generate_random_weights()
            else:
                version = latest_weights.get('version', 0) + 1
                logging.info(f"The current version is L2: {version}")
                global_model = np.array(latest_weights.get('model_weights', []))
                logging.info("Fetched the latest weights from the db")
            #global_model=Model(request.model_type).generate_random_weights()
            model_weights_array = np.array(model_weights)
            start_time = time.time()
            fedavg=FedAvg(global_model)
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
                'emulated_clients': "Yes",
                "layer": "L2",
                'timestamp': datetime.now()
            }
            # Insert the document into the collection
            #results=await collection.insert_one(document)
            await self.fetch_latest_weights("insert",document)
            #logging.info(f"Model sent successfully. Document ID: {results.inserted_id}")
            
            # Send the model to the L3 server, changed it here because it will be running inside the same container
            async with grpc.aio.insecure_channel('localhost:50052') as channel:
                stub = spotlight_pb2_grpc.ModelServiceStub(channel)
                try:
                    #logging.info(f"Sending the model with the type {type(request.model_type)}.")
                    grpc_request=spotlight_pb2.WeightsUpdate(
                        model_weights=model_weights_list,
                        num_samples=num_samples,
                        model_type=str(request.model_type)
                    )
                    await stub.updateL3(grpc_request)
                    logging.info("Sent the model to the L3 server.")
                except Exception as e:
                    logging.error(f"Error sending the model to the L3 server: {e}")
                await channel.close()
            
            return spotlight_pb2.UpdateResponse(message="Model sent successfully")
        except Exception as e:
            logging.error(f"Error sending the model: {e}")
            context.set_details(f"Error sending the model: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            return spotlight_pb2.UpdateResponse(message="Error sending the model")

# Create a gRPC server
class StartServer:
    def __init__(self):
        pass
    async def server(self):
        server = grpc.aio.server( options=[
                ('grpc.max_send_message_length', 500 * 1024 * 1024),  # 50 MB
                ('grpc.max_receive_message_length', 500 * 1024 * 1024)  # 50 MB
            ])
        l2server = L2server()
        spotlight_pb2_grpc.add_ModelServiceServicer_to_server(l2server, server)
        await l2server.startup_tasks()
        server.add_insecure_port('0.0.0.0:50051')
        await server.start()
        logging.info("Server started at port 50051 .")
        logging.info("Server started.")
        await server.wait_for_termination()          

# if __name__ == '__main__':
#     asyncio.run(server())