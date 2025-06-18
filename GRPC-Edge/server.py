import grpc
import torch
import torch.nn as nn
import torch.optim as optim
from concurrent import futures
import logging
from datetime import datetime
from Algorithms.fedavg import fed_avg
from pymongo import MongoClient
import edge_pb2
import edge_pb2_grpc
from celery import Celery
from edge_celery_tasks import send_model_to_l2_async, send_model_to_l3_async, store_aggregated_weights
import time
import io
import os
import bz2
import zipfile
import numpy as np
from models import SimpleCNN, TwoCNN, LogisticRegression, MobileNet, ResNet, SqueezeNet, DenseNetModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app_celery = Celery(
    "edge_celery_tasks",
    backend="redis://127.0.0.1:6381/0",
    broker="redis://127.0.0.1:6381/0",
)

model_classes = {
    "simplecnn": SimpleCNN,
    "twocnn": TwoCNN,
    "logistic_regression": LogisticRegression,
    "mobilenet": MobileNet,
    "resnet": ResNet,
    "squeezenet": SqueezeNet,
    "densenet": DenseNetModel,
}
compressed_models = ["resnet", "densenet", "squeezenet", "mobilenet"]

current_version = 1
aggregation_counter = 0
client_updates = []

try:
    mongo_client = MongoClient("mongodb://clUser:CloudLab@172.22.85.17:27017/")
    db = mongo_client["model_updates"]
    spot_db = mongo_client["emulator-data"]
    spot_collection = spot_db["layers"]
except Exception as e:
    logging.error(f"Error connecting to MongoDB: {e}")

def fetch_latest_weights(model_name):
    """Fetches the latest weights for a model and reconstructs them properly."""
    global use_latest_weights, current_version

    try:
        if not model_name or model_name.lower() not in model_classes:
            logging.error(f"Invalid model name received: {model_name}")
            return model_classes[model_name](), current_version

        model_name = model_name.lower().replace(" ", "_")
        collection = db[f"{model_name}_weights"]

        if model_name in compressed_models:
            latest_entry = collection.find_one({"model_name": model_name}, sort=[("timestamp", -1)])
            if not latest_entry:
                logging.warning(f"No valid weights found for {model_name}. Using default model.")
                return model_classes[model_name](), current_version

            latest_batch_id = latest_entry["batch_id"]
            latest_entries = list(collection.find(
                {"batch_id": latest_batch_id},
                sort=[("chunk_index", 1)]
            ))

            if not latest_entries:
                logging.error(f"No chunks found for batch_id: {latest_batch_id}")
                return model_classes[model_name](), current_version

            total_chunks = len(latest_entries)
            logging.info(f"Reconstructing model weights from {total_chunks} chunks for {model_name} (Batch ID: {latest_batch_id})")

            chunk_sizes = [len(chunk["model_weights"]) for chunk in latest_entries]
            logging.info(f"Chunk sizes: {chunk_sizes} | Total size before joining: {sum(chunk_sizes)} bytes")

            full_compressed_weights = b"".join(chunk["model_weights"] for chunk in latest_entries)

            try:
                decompressed_bytes = bz2.decompress(full_compressed_weights)
            except Exception as e:
                logging.error(f"BZ2 decompression failed: {e}")
                return model_classes[model_name](), current_version

            buffer = io.BytesIO(decompressed_bytes)
            buffer.seek(0)

            try:
                model_weights_data = torch.load(buffer, map_location="cpu")
                if isinstance(model_weights_data, bytes):
                    buffer.seek(0)
                    model_weights_data = torch.load(io.BytesIO(model_weights_data), map_location="cpu")
            except Exception as e:
                logging.error(f"Error loading PyTorch model: {e}")
                return model_classes[model_name](), current_version

            model = model_classes[model_name]()
            model.load_state_dict(model_weights_data)
            logging.info(f"Loaded latest model weights for {model_name} from MongoDB.")
            return model, latest_entries[0]["version"]

        else:
            latest_entry = collection.find_one({}, sort=[("DateTime", -1)])
            if not latest_entry or "model_weights" not in latest_entry:
                logging.warning(f"No valid weights found for {model_name}. Using default model.")
                return model_classes[model_name](), current_version

            float_list = latest_entry["model_weights"]
            if not float_list or not isinstance(float_list, list):
                logging.error(f"Invalid float list for {model_name}")
                return model_classes[model_name](), current_version

            float_array = np.array(float_list, dtype=np.float32)
            model = model_classes[model_name]()
            state_dict = model.state_dict()

            reshaped_state_dict = {}
            index = 0
            for key, param in state_dict.items():
                num_elements = param.numel()
                reshaped_tensor = torch.tensor(
                    float_array[index: index + num_elements],
                    dtype=param.dtype
                ).reshape(param.shape)
                reshaped_state_dict[key] = reshaped_tensor
                index += num_elements

            model.load_state_dict(reshaped_state_dict)
            logging.info(f"Reconstructed and loaded small model weights for {model_name} from MongoDB.")
            return model, latest_entry.get("version", current_version)

    except Exception as e:
        logging.error(f"Error fetching weights for {model_name} from MongoDB: {e}")

    return model_classes[model_name](), current_version

def is_spot_instance_available():
    try:
        spot_instance_status = spot_collection.find_one({"Layer": "l2"}, sort=[("timestamp", -1)])
        if spot_instance_status:
            status = spot_instance_status.get("Kill_signal", True)
            timestamp = spot_instance_status.get("timestamp", "No timestamp found")
            logging.info(
                f"Spot instance status: Layer={spot_instance_status['Layer']}, Kill_signal={status}, Timestamp={timestamp}"
            )
            if isinstance(status, bool) and not status:
                return True
            elif isinstance(status, str) and status.lower() == "false":
                return True
        logging.warning("No spot instance status found.")
        return False
    except Exception as e:
        logging.error(f"Error checking spot instance status: {e}")
        return False


# Define the EdgeLayerAggregator gRPC service
class EdgeLayerAggregator(edge_pb2_grpc.EdgeLayerAggregatorServiceServicer):
    def __init__(self):
        self.aggregated_weights = []
        self.version = 0
        self.aggregation_counter = 0

    def AggregateAndUpdateModel(self, request, context):
        global aggregation_counter, current_version
        request_counter = 0
        request_counter += 1
        received_version = request.version
        logging.info(f"Received aggregated weights with version: {request.version}")

        model_name = request.model_name.strip().lower()
        if not model_name or model_name not in model_classes:
            logging.error(f"Unknown or missing model name: {model_name}")
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            return edge_pb2.Ack(success=False, message=f"Invalid model name: {model_name}")

        model, latest_version = fetch_latest_weights(model_name)

        if received_version >= latest_version:
            current_version = received_version + 1 
        else:
            logging.warning(f"Received version {received_version} is behind stored version {latest_version}. Using latest version {latest_version}.")
            current_version = latest_version + 1

        try:
            if model_name in compressed_models:
                decompressed_bytes = bz2.decompress(request.weights)
                buffer = io.BytesIO(decompressed_bytes)
                buffer.seek(0)
                received_state_dict = torch.load(buffer, map_location="cpu")
                logging.info(f"Decompressed weights for {model_name}")
            else:
                model_state_dict = model.state_dict()
                weight_array = np.frombuffer(request.weights, dtype=np.float32)
                reshaped_weights = {}
                weight_index = 0
                for key, param in model_state_dict.items():
                    num_elements = param.numel()
                    reshaped_weights[key] = torch.tensor(
                        weight_array[weight_index:weight_index + num_elements]
                    ).reshape(param.shape)
                    weight_index += num_elements
                received_state_dict = reshaped_weights
                logging.info(f"Reconstructed raw weights from bytes for {model_name}")
        except Exception as e:
            logging.error(f"Failed to process received weights: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            return edge_pb2.Ack(success=False, message="Failed to load weights")

        start = time.time()
        updated_state_dict = fed_avg(model, [received_state_dict])
        end = time.time()
        aggregation_time = end - start

        if model_name in compressed_models:
            buffer = io.BytesIO()
            torch.save(updated_state_dict, buffer, _use_new_zipfile_serialization=False)
            compressed_weights = bz2.compress(buffer.getvalue())
        else:
            weight_tensor = []
            for _, value in updated_state_dict.items():
                weight_tensor.extend(value.flatten().tolist())
            compressed_weights = np.array(weight_tensor, dtype=np.float32).tobytes()
        
        layer = "Edge"
        updated_model_weights_payload = {
            "Layer": layer,
            "model_weights": compressed_weights,
            "version": current_version,
            "DateTime": datetime.utcnow(),
            "compressed": model_name in compressed_models
        }
        app_celery.send_task("edge_celery_tasks.store_aggregated_weights_task", args=[updated_model_weights_payload, model_name])

        model.load_state_dict(updated_state_dict)
        logging.info(f"Updated {model_name} model to version {current_version}")

        weights_payload = {
            "model_name": model_name,
            "model_weights": compressed_weights,
            "num_samples": request.num_samples,
            "version": current_version,
        }

        try:
            if is_spot_instance_available():
                logging.info(f"Spot L2 available. Sending {model_name} to L2.")
                send_model_to_l2_async.apply_async((weights_payload,))
            else:
                logging.info(f"Spot L2 unavailable. Sending {model_name} to Top Layer.")
                send_model_to_l3_async.apply_async((weights_payload,))
        except Exception as e:
            logging.error(f"Failed to forward weights: {e}")

        return edge_pb2.Ack(success=True, message="Aggregation done and model forwarded.")


def serve():
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ('grpc.max_send_message_length', 100 * 1024 * 1024),
            ('grpc.max_receive_message_length', 100 * 1024 * 1024)
        ]
    )
    edge_pb2_grpc.add_EdgeLayerAggregatorServiceServicer_to_server(EdgeLayerAggregator(), server)
    server.add_insecure_port("[::]:8083")
    logging.info("Edge layer started on port 8083.")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
