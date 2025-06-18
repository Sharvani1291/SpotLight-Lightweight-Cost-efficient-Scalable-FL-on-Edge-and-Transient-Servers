import grpc
import torch
import torch.nn as nn
import torch.optim as optim
from concurrent import futures
import logging
from datetime import datetime
from Algorithms.fedavg import fed_avg
from pymongo import MongoClient
import top_pb2
import top_pb2_grpc
import spotlight_pb2
import spotlight_pb2_grpc
from celery import Celery
from l3_celery_tasks import send_model_to_l2_async, send_model_to_edge_async, store_aggregated_weights
import time
import bz2
import io
import base64
import numpy as np
from models import SimpleCNN, TwoCNN, LogisticRegression, MobileNet, ResNet, SqueezeNet, DenseNetModel


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize Celery
app_celery = Celery(
    "l3_celery_tasks",
    backend="redis://127.0.0.1:6380/0",
    broker="redis://127.0.0.1:6380/0",
)

# Dictionary to dynamically select model
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


# MongoDB connection
try:
    mongo_client = MongoClient("mongodb://clUser:CloudLab@172.22.85.17:27017/")
    db = mongo_client["model_updates"]
    spot_db = mongo_client["emulator-data"]
    spot_collection = spot_db["layers"]
except Exception as e:
    logging.error(f"Error connecting to MongoDB: {e}")
    collection = None

def fetch_latest_weights(model_name):
    """Fetches the latest weights for a model and reconstructs them properly."""
    global use_latest_weights, current_version

    try:
        if not model_name or model_name.lower() not in model_classes:
            logging.error(f"Invalid model name received: {model_name}")
            return model_classes[model_name](), current_version

        model_name = model_name.lower().replace(" ", "_")
        collection = db[f"{model_name}_weights"]

        # For large compressed models
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

        # For small models (uncompressed float list)
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

# Function to check if the spot instance is available
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


# Define the TopLayerAggregator gRPC service
class TopLayerAggregator(top_pb2_grpc.TopLayerAggregatorServiceServicer):
    
    def ReceiveAggregatedWeightsFromEdge(self, request, context):
        """
        Receives aggregated model weights from Edge, decompresses or reconstructs, and updates the global model.
        """
        global current_version

        try:
            received_version = request.version
            model_name = request.model_name.strip().lower()
            logging.info(f"Received aggregated weights from Edge for {model_name}, version {received_version}")

            if not model_name or model_name not in model_classes:
                logging.error(f"Invalid model name: {model_name}")
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                return spotlight_pb2.Ack(success=False, message=f"Invalid model: {model_name}")

            current_version = max(received_version, current_version) + 1

            if model_name in compressed_models:
                received_weights = request.weights
                logging.info(f"First 20 bytes of compressed weights: {received_weights[:20]}")
                logging.info(f"Size of compressed weights: {len(received_weights)} bytes")
                
                try:
                    decompressed_bytes = bz2.decompress(received_weights)
                    buffer = io.BytesIO(decompressed_bytes)
                    received_state_dict = torch.load(buffer, map_location="cpu")
                    logging.info(f"Decompressed and loaded state_dict for {model_name}")
                except Exception as e:
                    logging.error(f"Error decompressing weights for {model_name}: {e}")
                    context.set_code(grpc.StatusCode.INTERNAL)
                    return spotlight_pb2.Ack(success=False, message="Decompression failed")
            else:
                float_list = list(request.float_weights)
                logging.info(f"First 20 float values: {float_list[:20]}")
                logging.info(f"Total floats received: {len(float_list)}")

                if len(float_list) == 0:
                    logging.error(f"Received empty float_weights for {model_name}")
                    context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                    return spotlight_pb2.Ack(success=False, message="Empty float_weights")

                try:
                    float_array = np.array(float_list, dtype=np.float32)
                    model_state_dict = model_classes[model_name]().state_dict()
                    reshaped_weights = {}
                    index = 0
                    for key, param in model_state_dict.items():
                        num_elements = param.numel()
                        reshaped_weights[key] = torch.tensor(
                            float_array[index:index + num_elements],
                            dtype=param.dtype
                        ).reshape(param.shape)
                        index += num_elements
                    received_state_dict = reshaped_weights
                    logging.info(f"Reconstructed state_dict from float_weights for {model_name}")
                except Exception as e:
                    logging.error(f"Error reconstructing float_weights: {e}")
                    context.set_code(grpc.StatusCode.INTERNAL)
                    return spotlight_pb2.Ack(success=False, message="Failed to process float_weights")

            model = model_classes[model_name]()
            model.load_state_dict(received_state_dict)
            logging.info(f"Updated model to version {current_version} for {model_name}")

            return spotlight_pb2.Ack(success=True, message="Weights from Edge received successfully")

        except Exception as e:
            logging.error(f"Error in ReceiveAggregatedWeightsFromEdge: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            return spotlight_pb2.Ack(success=False, message=f"Error: {str(e)}")

    def SendAggregatedWeights(self, request, context):
        global current_version, aggregation_counter, client_updates

        received_version = request.version
        model_name = request.model_name.lower()
        num_samples = request.num_samples
        received_weights = request.weights

        logging.info(f"Received aggregated weights for {model_name}, version {received_version}")

        if model_name not in model_classes:
            logging.error(f"Unknown model name: {model_name}")
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            return top_pb2.Ack(success=False, message=f"Invalid model: {model_name}")

        try:
            if model_name in compressed_models:
                decompressed_bytes = bz2.decompress(received_weights)
                buffer = io.BytesIO(decompressed_bytes)
                buffer.seek(0)
                received_state_dict = torch.load(buffer, map_location="cpu")
            else:
                float_array = np.frombuffer(received_weights, dtype=np.float32).tolist()
                dummy_model = model_classes[model_name]()
                reshaped_weights = {}
                weight_index = 0
                for key, param in dummy_model.state_dict().items():
                    num_elements = param.numel()
                    reshaped_weights[key] = torch.tensor(
                        float_array[weight_index:weight_index + num_elements],
                        dtype=torch.float32
                    ).reshape(param.shape)
                    weight_index += num_elements
                received_state_dict = reshaped_weights

            logging.info(f"Parsed incoming weights for {model_name}")
        except Exception as e:
            logging.error(f"Error processing weights for {model_name}: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            return top_pb2.Ack(success=False, message="Weight parsing failed")

        client_updates.append((received_state_dict, num_samples))
        aggregation_counter += 1
        logging.info(f"Aggregation counter: {aggregation_counter}/3")

        if aggregation_counter < 3:
            return top_pb2.Ack(success=True, message="Weights received and stored. Waiting for more updates.")

        model, latest_version = fetch_latest_weights(model_name)
        current_version = max(received_version, latest_version) + 1
        aggregated_weights = fed_avg(model, [update[0] for update in client_updates])
        model.load_state_dict(aggregated_weights)
        logging.info(f"Performed FedAvg for {model_name}, version {current_version}")

        aggregation_counter = 0
        client_updates.clear()

        if model_name in compressed_models:
            buffer = io.BytesIO()
            torch.save(model.state_dict(), buffer)
            compressed_weights = bz2.compress(buffer.getvalue())
        else:
            compressed_weights = []
            for _, param in model.state_dict().items():
                compressed_weights.extend(param.flatten().tolist())

        payload = {
            "Layer": 3,
            "model_weights": compressed_weights,
            "version": current_version,
            "DateTime": datetime.utcnow(),
            "compressed": model_name in compressed_models
        }
        app_celery.send_task("l3_celery_tasks.store_aggregated_weights_task", args=[payload, model_name])

        weights_payload = {
            "model_name": model_name,
            "model_weights": compressed_weights,
            "num_samples": num_samples,
            "version": current_version
        }

        try:
            if is_spot_instance_available():
                logging.info(f"Sending {model_name} to L2.")
                send_model_to_l2_async.apply_async((weights_payload,))
            else:
                logging.info(f"Spot not available. Sending {model_name} to Edge.")
                send_model_to_edge_async.apply_async((weights_payload,))
        except Exception as e:
            logging.error(f"Failed to forward weights: {e}")

        return top_pb2.Ack(success=True, message="Model aggregated and forwarded successfully.")


        def _flatten_weights(self, state_dict):
            """Flatten model weights into a single list of floats."""
            flattened_weights = []
            for key, value in state_dict.items():
                flattened_weights.extend(value.cpu().numpy().flatten().tolist())
            return flattened_weights

def serve():
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ('grpc.max_send_message_length', 500 * 1024 * 1024),
            ('grpc.max_receive_message_length', 500 * 1024 * 1024)
        ]
    )
    top_pb2_grpc.add_TopLayerAggregatorServiceServicer_to_server(TopLayerAggregator(), server)
    server.add_insecure_port("[::]:8082")
    logging.info("Top layer aggregator server started on port 8082.")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()