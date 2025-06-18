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
from bson.binary import Binary
import uvloop
#Useraddd modules
from modelGenerator import Model
#from Algorithms.NumbaFedBuff import NumbaFedBuff
#from Algorithms.FedProx import FedProx
#from Algorithms.FedAvg import FedAvg
from Algorithms.FedAdam import FedAdam
# Set up logging to display information about the process
#logging.basicConfig(level=logging.INFO)


logging.basicConfig(filename='L2.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

logging.info("Starting the GRPC client script.")
#use .env file to store the connection string

class L2server(spotlight_pb2_grpc.ModelServiceServicer):
    def __init__(self):
        #loads the env variables
        load_dotenv()
        self.uri=os.getenv("MONGO_URI")
        self.db=os.getenv("MONGO_DB")
        self.collection=os.getenv("MONGO_COLLECTION")
        self.client = AsyncIOMotorClient(self.uri, maxPoolSize=20, minPoolSize=10)
        # Select the database and collection
        self.db = self.client[self.db]
        self.collection = self.db[self.collection]
        self.collection.create_index("timestamp")
        self.model_list=[]
        self.sample_counts=[]
        self.model_weights_list=[]
        self.aggregation_lock = asyncio.Lock()
        self.latest_weights=None

        init_model = Model(os.getenv("model", "cnn"), dtype=np.float64,target_mb=10).generate_random_weights()
        self.layerId=uuid.uuid4()
        self.optimizer = FedAdam(init_model,use_amsgrad=True,keep_ema=0.999,allocate_tmp=True,clip_threshold=1.0)  
        #self.optimizer = FedProx(init_model, mu=0.7)

        self.collection.update_one(
        {"_id": "version_counter"},
        {"$setOnInsert": {"version": 0}},
        upsert=True
)
    
    #This function fetches the latest weights from the database and inserts the new weights,I made one function to create just one pool
    async def fetch_latest_weights(self,function_term,payload=None):
        try:
        # Connect to the MongoDB cluster
        
            if function_term=="latest":
            # Find the latest document in the collection
                #latest_weights =  await self.collection.find_one(
    #{},
    #sort=[("timestamp", -1)],
    #projection={"model_weights": 1, "version": 1, "_id": 0}
#)
                logging.info("Fetched the latest weights from the db")
                # Return the weights
                return self.latest_weights
            if function_term=="insert":
                result=await self.collection.insert_one(payload)
                logging.info(f"Model sent successfully. Document ID: {result.inserted_id}")
        except Exception as e:
            logging.error(f"Error fetching the latest weights: {e}")
            raise 
    async def GetModel(self, request, context):
        logging.info("Received a request for the model.")
        try:
            #commeted it out because I need speed
            #await self.fetch_latest_weights("latest")
            return Empty()
        except Exception as e:
            logging.error(f"Error fetching the latest weights from the Get method: {e}")
            context.set_details(f"Error fetching the latest weights from the get Method: {e}")
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
                #global_model=Model(os.getenv("model", "cnn"), dtype=np.float64,target_mb=10).generate_random_weights()
            else:
                #version = latest_weights.get('version', 0) + 1
                version=latest_weights.get('version', 0) 
                logging.info(f"The current version is L2: {version}")
                #global_model = np.array(latest_weights.get('model_weights', []))
                logging.info("Fetched the latest weights from the db")
            #global_model=Model(request.model_type).generate_random_weights()
            model_weights_array = np.array(model_weights)
            
            self.model_list.append(model_weights_array)
            self.sample_counts.append(num_samples)
            #logging.info(f"Number of models collected: {len(self.model_list)}")
            #logging.info(f"Sample counts: {self.sample_counts}")
            #logging.info(f"First model shape: {np.array(self.model_list[0]).shape if self.model_list else 'N/A'}")
            loop = asyncio.get_running_loop()
#updated_global_model = await loop.run_in_executor(None, self.optimizer.aggregate, self.model_list, self.sample_counts)
            if len(self.model_list) >= 10:
                async with self.aggregation_lock:
                    if len(self.model_list) >= 1:
                        version = version + 1
                        start_time = time.time()    
                        #fedAdam=FedAdam(global_model)
                        updated_global_model = await loop.run_in_executor(None, self.optimizer.aggregate, self.model_list, self.sample_counts)
                        #updated_global_model=self.optimizer.aggregate(self.model_list,self.sample_counts)
                        # logging.info(
                        #     "FedAdam state → t=%d, ‖m‖=%.3e, ‖v‖=%.3e",
                        #     self.optimizer.t,
                        #     np.linalg.norm(self.optimizer.m),
                        #     np.linalg.norm(self.optimizer.v),
                        # )
                        self.model_list.clear()
                        self.sample_counts.clear()
                        end_time = time.time()
                        self.request_counter = 0
                        aggregation_time=end_time-start_time
                        logging.info(f"Time taken to aggregate the model weights: {aggregation_time} seconds")
                        logging.info("Aggregated the model weights.")
                        #convert the updated global model to a format suitable for MongoDB
                        weights_f32 = updated_global_model.astype(np.float32, copy=False)
                        logging.info(f"Updated global model type: {weights_f32.dtype}")
                        #self.model_weights_list=weights_f32.tolist()
                        
                        
                        # Convert the model weights to a format suitable for MongoDB
                        blob = Binary(weights_f32.tobytes())
                        #added this to ensure that we go to the next version only after the aggregation is done
                        #version=version + 1
                        logging.info(f"The new version of the model: {version}")
                        #version = latest_weights.get('version', 0) + 1
                        document = {
                            '_id': uuid.uuid4().hex,
                            'model_weights': blob,
                            "LayerId": str(self.layerId),
                            'aggregation_time': aggregation_time,
                            'version': version,
                            'number_of_samples': num_samples,
                            'emulated_clients': "Yes",
                            "layer": "L2",
                            'timestamp': datetime.now()
                        }
                        # Insert the document into the collection
                        #results=await collection.insert_one(document)
                        logging.debug("Document to be inserted")
                        await self.fetch_latest_weights("insert",document)
                        #logging.info(f"Model sent successfully. Document ID: {results.inserted_id}")
                        #logging.info(self.model_weights_list)
                        # Send the model to the L3 server
                        async with grpc.aio.insecure_channel('172.22.86.230:50052',options=[
            ('grpc.max_send_message_length',     100 * 1024 * 1024),  # 100 MiB
            ('grpc.max_receive_message_length',  100 * 1024 * 1024),  # 100 MiB
        ]) as channel:
                            stub = spotlight_pb2_grpc.ModelServiceStub(channel)
                            try:
                                logging.info(f"Sending the model with the type {type(request.model_type)}.")
                                grpc_request=spotlight_pb2.WeightsUpdate(
                                    model_weights=updated_global_model,
                                    num_samples=num_samples,
                                    model_type=str(request.model_type)
                                )
                                await stub.updateL3(grpc_request)
                                logging.info("Sent the model to the L3 server.")
                            except Exception as e:
                                logging.error(f"Error sending the model to the L3 server: {e}")
                            await channel.close()
                        
                        return spotlight_pb2.UpdateResponse(message="Model sent successfully")
            else:
                
                logging.info("Not enough models to aggregate. Waiting for more models.")
                return spotlight_pb2.UpdateResponse(message="Waiting for more models")
        except Exception as e:
            logging.error(f"Error sending the model in the update method: {e}")
            context.set_details(f"Error sending the model in the update method: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            return spotlight_pb2.UpdateResponse(message="Error sending the model")

# Create a gRPC server
async def server():
    try:
        server = grpc.aio.server( options=[
                ('grpc.max_send_message_length', 500 * 1024 * 1024),  # 50 MB
                ('grpc.max_receive_message_length', 500 * 1024 * 1024),
                # 50 MB
            ])
        spotlight_pb2_grpc.add_ModelServiceServicer_to_server(L2server(), server)
        server.add_insecure_port('[::]:50051')
        await server.start()
        logging.info("Server started at port 50051 .")
        logging.info("Server started.")
        await server.wait_for_termination()
    except Exception as e:
        logging.error(f"Error starting the server: {e}")
        raise          

if __name__ == '__main__':
    
    asyncio.run(server())
