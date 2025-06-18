import grpc
import logging
import numpy as np
from datetime import datetime
from time import sleep
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import spotlight_pb2
import spotlight_pb2_grpc
import csv
import threading
import numba
# Logging Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
    logging.FileHandler("server_emulated.log"),
    logging.StreamHandler()
])

# Global Model (NumPy Array)
aggregation_goal = 5
current_model_version = 0
client_updates = []
client_id_counter = 0
request_counter = 0
global_model = np.random.rand(109184)# Simulating a large model with random weights
client_count=0
aggregation_time=0
total_aggregation_time=0
lock = threading.Lock()

def fetch_latest_weights():
    try:
        logging.info("Fetching latest weights from MongoDB...")
        sleep(5)  # Simulate delay
    except Exception as e:
        logging.error(f"Error connecting to MongoDB: {e}")

# FedBuff Aggregation using NumPy
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

    logging.info("Aggregation process completed.")
    return aggregated_weights

def update_global_model(aggregated_weights):
    global global_model, current_model_version
    global_model = aggregated_weights
    sleep(5)# Update the global model with aggregated weights
    current_model_version += 1
    logging.info(f"Global model updated to version {current_model_version}.")
    


@njit
def _aggregate_weights_numba(weights_array, samples_array, staleness_array):
    num_clients = len(weights_array)
    weight_shape = weights_array[0].shape
    aggregated_weights = np.zeros(weight_shape)
    total_weighted_samples = 0.0

    for i in range(num_clients):
        weights = weights_array[i]
        num_samples = samples_array[i]
        staleness = staleness_array[i]
        staleness_factor = 1.0 / (1.0 + staleness)
        weighted_weights = weights * (num_samples * staleness_factor)
        aggregated_weights += weighted_weights
        total_weighted_samples += num_samples * staleness_factor

    aggregated_weights /= total_weighted_samples
    return aggregated_weights

def numba_fedbuff_aggregate(client_updates):
    """
    client_updates: list of tuples (weights: np.ndarray, num_samples: int, staleness: float)
    """
    logging.info("Starting FedBuff aggregation...")

    weights_array = []
    samples_array = []
    staleness_array = []

    for weights, num_samples, staleness in client_updates:
        logging.info(f"Client update: Num Samples = {num_samples}, Staleness = {staleness}")
        weights_array.append(np.array(weights))
        samples_array.append(float(num_samples))
        staleness_array.append(float(staleness))

    aggregated_weights = _aggregate_weights_numba(weights_array, np.array(samples_array), np.array(staleness_array))
    return aggregated_weights


# FederatedLearningService Implementation
class FederatedLearningService(spotlight_pb2_grpc.FederatedLearningServiceServicer):
    
    def registerClient(self, request, context):
        global client_count, request_counter
        if request.clientId:
            logging.info(f"Client {request.clientId} registered.")
            with lock:
                client_count+=1
                request_counter+=1
        return spotlight_pb2.Ack(success=True, message="Client registered successfully.")
        
    def GetGlobalModel(self, request, context):
        global global_model, current_model_version,request_counter
        with lock:
            request_counter += 1
        fetch_latest_weights()
        global_weights = global_model.flatten().tolist()  # Flatten the model for transmission
        return spotlight_pb2.ModelResponse(weights=global_weights, version=str(current_model_version))

    def SendModelWeights(self, request, context):
        global client_updates, client_id_counter, current_model_version, request_counter,aggregation_time,total_aggregation_time,aggregation_goal,client_count
        # Increment Client ID Counter
        client_id=request.clientId
        #appends the clientId to the list
        with lock:
            request_counter += 1

        logging.info(f"Received update from client {client_id} with version {request.version}.")
        client_weights = np.array(request.weights)  # NumPy array for weights
        num_samples = request.num_samples
        client_version = int(request.version)
        staleness = current_model_version - client_version
        logging.info(f"Client {client_id}: Calculated staleness = {staleness}")

        # Handle Outdated Client Versions 
        if client_version < current_model_version:
            logging.info(f"Client version {client_version} is outdated. Fetching latest weights.")
            fetch_latest_weights()

        # Add Client Update to Buffer
        with lock:
            client_updates.append((client_id, client_weights, num_samples, staleness))

        # Perform Aggregation if Goal is Met
        if len(client_updates) >= aggregation_goal:
            logging.info("Aggregation goal reached. Performing FedBuff aggregation.")
            start_time = datetime.utcnow()
            #aggregated_weights = fedbuff_aggregate(client_updates)
            aggregated_weights = numba_fedbuff_aggregate(client_updates)
            logging.info(f"Aggregated weights: {aggregated_weights}")
            update_global_model(aggregated_weights)
            end_time = datetime.utcnow()
            aggregation_time=(end_time-start_time).total_seconds()
            #adds all the aggregation time
            total_aggregation_time+=aggregation_time
            # aggregation_timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
            
            # aggregated_data = {
            #     "aggregated_weights": aggregated_weights.tolist(),
            #     "model_name": "Papaya",
            #     "request_count": request_counter,
            #     "aggregation_time": total_aggregation_time
            # }
            #required_request=aggregation_goal*7
            
            # Clear client updates after aggregation
            logging.info(f"Current request count: {request_counter}")
            server_end_time=datetime.utcnow()
            total_process_time=(server_end_time-server_start).total_seconds()
            logging.info("Required request count met. Writing to CSV file.")
            with lock:
                try:
                    with open('aggregation_results.csv', mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([client_count, request_counter, total_aggregation_time,total_process_time])
                except Exception as e:
                    logging.error(f"Error writing to CSV file: {e}")
            client_updates.clear()
                    #increases the aggregation goal by 100, resets the total aggregation time,request counter is set to 0
        aggregation_goal+=100
        total_aggregation_time=0
        request_counter=0
        client_count=0

        return spotlight_pb2.Ack(success=True, message="Weights received and processed.")   

# Start gRPC Server
def emulated_server():
    server = grpc.server(ThreadPoolExecutor(max_workers=4),options=[
    ('grpc.max_message_length', 100 * 1024 * 1024),  # 50 MB
])
    spotlight_pb2_grpc.add_FederatedLearningServiceServicer_to_server(FederatedLearningService(), server)
    server.add_insecure_port("[::]:8088")
    logging.info("Server started on port 8088.")
    server.start()
    server.wait_for_termination()

if __name__ == "__main__":
    server_start=datetime.utcnow()
    with open('aggregation_results.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Number of Clients', 'Number of Requests', 'Aggregation Time','Total Processing Time'])
    emulated_server()
    
