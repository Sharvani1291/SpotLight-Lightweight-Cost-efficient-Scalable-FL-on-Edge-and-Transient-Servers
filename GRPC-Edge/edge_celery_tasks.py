from celery import Celery
from pymongo import MongoClient
import logging
import grpc
import torch
from datetime import datetime
from bson.objectid import ObjectId
from edge_pb2 import ModelState
from edge_pb2 import ModelWeights
from edge_pb2 import AggregatedWeights
from edge_pb2_grpc import EdgeLayerAggregatorServiceStub
from top_pb2_grpc import TopLayerAggregatorServiceStub
from edge_pb2 import AggregatedWeights
from spotlight_pb2_grpc import ModelServiceStub
import torch.nn as nn
import logging
import copy
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'GRPC-Server')))

MAX_MONGO_SIZE = 15 * 1024 * 1024

compressed_models = ["resnet", "densenet", "squeezenet", "mobilenet"]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

app = Celery('edge_celery_tasks', backend='redis://127.0.0.1:6381/0', broker='redis://127.0.0.1:6381/0')
app.autodiscover_tasks(['edge_celery_tasks'])

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    
def store_aggregated_weights(model_weights, model_name):
    """Stores aggregated weights in MongoDB with proper PyTorch serialization and version tracking."""
    try:
        mongo_client = MongoClient("mongodb://clUser:CloudLab@172.22.85.17:27017/")
        db = mongo_client["model_updates"]
        collection = db[f"{model_name}_weights"]

        batch_id = f"{model_name}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        version = model_weights.get("version", 1)

        logging.info(f"🗜 Storing weights for {model_name} (Batch ID: {batch_id}, Version: {version})")

        if model_name in compressed_models:
            compressed_weights = model_weights["model_weights"]

            logging.info(f"Compressed weights size: {len(compressed_weights)} bytes")
            logging.info(f"First 20 bytes of compressed weights: {compressed_weights[:20]}")

            # Store in chunks if size > 16MB
            if len(compressed_weights) > MAX_MONGO_SIZE:
                logging.info(f"Splitting large model weights for {model_name} into multiple chunks.")

                chunks = []
                for i in range(0, len(compressed_weights), MAX_MONGO_SIZE):
                    chunk = compressed_weights[i: i + MAX_MONGO_SIZE]
                    chunks.append(chunk)

                # Store each chunk separately
                for i, chunk in enumerate(chunks):
                    logging.info(f"Storing chunk {i}/{len(chunks)} | Size: {len(chunk)} bytes")

                    collection.insert_one({
                        "_id": f"{batch_id}_{i}",
                        "batch_id": batch_id,
                        "model_name": model_name,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "model_weights": chunk,
                        "compressed": True,
                        "version": version,
                        "timestamp": datetime.utcnow()
                    })

                logging.info(f"Stored model weights in {len(chunks)} chunks for {model_name} (Batch ID: {batch_id}, Version: {version}).")
                return f"Stored in {len(chunks)} chunks (Batch ID: {batch_id}, Version: {version})"

            else:
                # Store as a single document if under 16MB
                logging.info(f"Storing model weights as a single document (size: {len(compressed_weights)} bytes).")
                result = collection.insert_one({
                    "_id": batch_id,
                    "batch_id": batch_id,
                    "model_name": model_name,
                    "model_weights": compressed_weights,
                    "compressed": True,
                    "version": version,
                    "timestamp": datetime.utcnow()
                })
                logging.info(f"Stored aggregated weights successfully. Inserted ID: {result.inserted_id}")
                return str(result.inserted_id)

        else:
            custom_id = f"{model_name}_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}"
            result = collection.insert_one({
                "_id": custom_id,
                "Layer": model_weights.get("Layer"),
                "num_samples": model_weights.get("num_samples"),
                "version": version,
                "Aggregation_time": model_weights.get("Aggregation_time"),
                "Client_training_time": model_weights.get("Client_training_time"),
                "DateTime": model_weights.get("DateTime"),
                "Number_of_requests": model_weights.get("Number_of_requests"),
                "model_weights": model_weights.get("model_weights")
            })

            logging.info(f"Stored aggregated weights successfully for small model {model_name}. Inserted ID: {result.inserted_id}")
            return str(result.inserted_id)

    except Exception as e:
        logging.error(f"Error inserting model weights: {e}", exc_info=True)
        return f"Error: {e}"

@app.task
def send_model_to_l2_async(weights_payload):
    try:
        model_name = weights_payload.get("model_name", "").strip().lower()
        model_weights = weights_payload.get("model_weights")
        num_samples = weights_payload.get("num_samples")
        version = weights_payload.get("version")

        if not model_name or num_samples is None or version is None:
            raise ValueError("Missing required fields in payload")

        logging.info(f"Preparing to send weights to L2 for model: {model_name}")

        if model_name in compressed_models:
            if not isinstance(model_weights, bytes):
                raise ValueError(f"Expected bytes for large model `{model_name}`, got {type(model_weights)}")
            payload = AggregatedWeights(
                model_name=model_name,
                weights=model_weights,
                num_samples=num_samples,
                version=version
            )
        else:
            if not isinstance(model_weights, bytes):
                raise ValueError(f"Expected bytes for small model `{model_name}`, got {type(model_weights)}")
            payload = AggregatedWeights(
                model_name=model_name,
                float_weights=model_weights,
                num_samples=num_samples,
                version=version
            )
            logging.info(f"Sending float_weights for small model {model_name}. Size: {len(model_weights)} bytes")

        channel = grpc.insecure_channel('localhost:50051')
        stub = ModelServiceStub(channel)
        response = stub.ReceiveAggregatedWeightsFromL3(payload)

        logging.info(f"Response from L2 server: {response}")
        return {"success": response.success, "message": response.message}

    except grpc.RpcError as grpc_error:
        logging.error(f"gRPC error: {grpc_error.details()}")
        raise
    except Exception as e:
        logging.error(f"Error in send_model_to_l2_async: {e}")
        raise


@app.task
def send_model_to_l3_async(updated_model_weights_l3):
    try:
        model_name = updated_model_weights_l3.get("model_name", "").strip().lower()
        model_weights = updated_model_weights_l3.get("model_weights")
        num_samples = updated_model_weights_l3.get("num_samples")
        version = updated_model_weights_l3.get("version")

        if not model_name or num_samples is None or version is None:
            raise ValueError("Missing required fields in payload")

        logging.info(f"Preparing to send weights to L3 for model: {model_name}")

        if model_name in compressed_models:
            if not isinstance(model_weights, bytes):
                raise ValueError(f"Expected bytes for large model `{model_name}`, got {type(model_weights)}")
            
            payload = AggregatedWeights(
                model_name=model_name,
                weights=model_weights,
                num_samples=num_samples,
                version=version
            )
            logging.info(f"Using pre-compressed weights for {model_name}")

        else:
            if isinstance(model_weights, list):
                model_weights = torch.tensor(model_weights, dtype=torch.float32).numpy().tobytes()
                logging.info(f"Converted float list to bytes for {model_name}. Size: {len(model_weights)} bytes")
            elif not isinstance(model_weights, bytes):
                raise ValueError(f"Expected list or bytes for small model `{model_name}`, got {type(model_weights)}")
            
            payload = AggregatedWeights(
                model_name=model_name,
                float_weights=model_weights,
                num_samples=num_samples,
                version=version
            )
            logging.info(f"Sending float_weights for small model {model_name}. Size: {len(model_weights)} bytes")

        channel = grpc.insecure_channel('localhost:8082')
        stub = TopLayerAggregatorServiceStub(channel)
        response = stub.ReceiveAggregatedWeightsFromEdge(payload)

        logging.info(f"Response from L3 server: {response}")
        return {"success": response.success, "message": response.message}

    except grpc.RpcError as grpc_error:
        logging.error(f"gRPC error: {grpc_error.details()}")
        raise
    except Exception as e:
        logging.error(f"Error in send_model_to_l3_async: {e}")
        raise
    
@app.task
def store_aggregated_weights_task(model_weights, model_name):
    if model_weights is None or model_name is None:
        raise ValueError("Both model_weights and model_name must be provided.")
    try:
        logger.info("store_aggregated_weights_task started")
        logger.info(f"Invoked store_aggregated_weights_task with model_name: {model_name}")
        
        result = store_aggregated_weights(model_weights, model_name)
        
        return result
    except Exception as e:
        logger.error(f"Error in store_aggregated_weights_task: {e}", exc_info=True)
        raise


