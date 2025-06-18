import grpc
import torch
import torch.nn as nn
from concurrent import futures
from celery import Celery
from pymongo import MongoClient
import logging
import spotlight_pb2
import spotlight_pb2_grpc
from celery_tasks import store_aggregated_weights_task
from datetime import datetime

# Logging Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global Variables
aggregation_goal = 2
current_model_version = 0
client_updates = []
client_id_counter = 0
request_counter = 0
global_model = None

# MongoDB Configuration( need to fix this with a seperate connection object/pool that connects easily )
try:
    mongo_client = MongoClient("mongodb://172.22.85.17:27017/")
    db = mongo_client["model_updates"]
    collection = db["Papaya_weights"]
    logging.info("MongoDB connection successful!")
except Exception as e:
    logging.error(f"Error connecting to MongoDB: {e}")

# Celery Configuration
app_celery = Celery(
    'celery_tasks',
    backend='redis://127.0.0.1:6381/0',
    broker='redis://127.0.0.1:6381/0'
)

# Define CNN Model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize the Global Model
global_model = SimpleCNN()

# Fetch Latest Weights from MongoDB
def fetch_latest_weights():
    global collection, global_model
    try:
        latest_entry = collection.find_one(sort=[("_id", -1)])  # Fetch latest MongoDB entry
        if latest_entry:
            # Access the nested 'aggregated_weights' inside 'model_weights'
            model_weights = latest_entry.get('model_weights', {}).get('aggregated_weights', None)
            latest_version = latest_entry.get('version', 0)

            if model_weights and isinstance(model_weights, list):  # Ensure weights are a list
                state_dict = global_model.state_dict()
                offset = 0
                # Iterate through model's state_dict and update weights
                for key, param in state_dict.items():
                    param_size = param.numel()
                    weight_slice = model_weights[offset:offset + param_size]
                    reshaped_weights = torch.tensor(weight_slice).reshape(param.shape)
                    param.data.copy_(reshaped_weights)
                    offset += param_size
                logging.info(f"Fetched latest weights from MongoDB.")
                return latest_version
            else:
                logging.error("Aggregated weights are missing or not in the correct list format.")
        else:
            logging.warning("No entries found in MongoDB.")
            return 0
    except Exception as e:
        logging.error(f"Error fetching weights from MongoDB: {e}")
        return 0

# FedBuff Aggregation
def fedbuff_aggregate(client_updates):
    global global_model
    total_weighted_samples = 0
    aggregated_weights = None

    logging.info("Starting aggregation process...")
    for i, (client_id, weights, num_samples, staleness) in enumerate(client_updates):
        # Adjust weights based on staleness
        staleness_factor = 1 / (1 + staleness)
        logging.info(f"Client {client_id}: Staleness = {staleness}, Staleness Factor = {staleness_factor}, Num Samples = {num_samples}")
        
        weighted_weights = weights * (num_samples * staleness_factor)

        if aggregated_weights is None:
            aggregated_weights = weighted_weights
        else:
            aggregated_weights += weighted_weights

        total_weighted_samples += num_samples * staleness_factor

    # Normalize by the total weighted samples
    aggregated_weights /= total_weighted_samples

    # Validate weights size
    total_model_parameters = sum(param.numel() for param in global_model.parameters())
    if aggregated_weights.numel() != total_model_parameters:
        raise ValueError(f"Aggregated weights size {aggregated_weights.numel()} " f"does not match total model parameters size {total_model_parameters}")

    logging.info("Aggregation process completed.")
    return aggregated_weights


# Update Global Model
def update_global_model(aggregated_weights):
    global global_model, current_model_version
    offset = 0
    state_dict = global_model.state_dict()

    for key, param in state_dict.items():
        param_size = param.numel()
        param_data = aggregated_weights[offset:offset + param_size].reshape(param.shape)
        param.data.copy_(param_data)
        offset += param_size

    current_model_version += 1
    logging.info(f"Global model updated to version {current_model_version}.")

# FederatedLearningService Implementation
class FederatedLearningService(spotlight_pb2_grpc.FederatedLearningServiceServicer):
    def GetGlobalModel(self, request, context):
        global global_model, current_model_version
        global_weights = []
        for param in global_model.parameters():
            global_weights.extend(param.data.cpu().numpy().flatten())
        return spotlight_pb2.ModelResponse(weights=global_weights, version=str(current_model_version))

    def SendModelWeights(self, request, context):
        global client_updates, client_id_counter, current_model_version, request_counter

        # Increment Client ID Counter
        client_id_counter += 1
        client_id = client_id_counter
        request_counter += 1

        logging.info(f"Received update from client {client_id} with version {request.version}.")
        client_weights = torch.tensor(request.weights, dtype=torch.float32)
        num_samples = request.num_samples
        client_version = int(request.version)
        staleness = current_model_version - client_version
        logging.info(f"Client {client_id}: Calculated staleness = {staleness}")

        # Handle Outdated Client Versions
        if client_version < current_model_version:
            logging.info(f"Client version {client_version} is outdated. Fetching latest weights.")
            fetch_latest_weights()

        # Add Client Update to Buffer
        client_updates.append((client_id, client_weights, num_samples, staleness))

        # Perform Aggregation if Goal is Met
        if len(client_updates) >= aggregation_goal:
            logging.info("Aggregation goal reached. Performing FedBuff aggregation.")
            aggregated_weights = fedbuff_aggregate(client_updates)
            update_global_model(aggregated_weights)
            
            aggregation_timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')

            aggregated_data = {
                "aggregated_weights": aggregated_weights.tolist(),
                "model_name": "Papaya",
                "request_count": request_counter,
                "aggregation_timestamp": aggregation_timestamp
            }
            store_aggregated_weights_task.delay(aggregated_data, "Papaya")

            client_updates.clear()

        return spotlight_pb2.Ack(success=True, message="Weights received and processed.")

# Start gRPC Server
def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    spotlight_pb2_grpc.add_FederatedLearningServiceServicer_to_server(FederatedLearningService(), server)
    server.add_insecure_port("[::]:8082")
    logging.info("Server started on port 8082.")
    server.start()
    server.wait_for_termination()

if __name__ == "__main__":
    serve()
