from fastapi import FastAPI, HTTPException, Body
from tasks import store_aggregated_weights_task
from celery import Celery
import celery_configuration
from pydantic import BaseModel
import torch.optim as optim
import torch.nn as nn
from pymongo import MongoClient
import torch
from Algorithms.fedavg import fed_avg
import logging
from typing import List, Dict, Tuple

logging.basicConfig(level=logging.INFO)

app = FastAPI()

app_celery = Celery('tasks', backend=celery_configuration.CELERY_RESULT_BACKEND, broker=celery_configuration.CELERY_BROKER_URL)
app_celery.config_from_object('celery_configuration')

try:
    mongo_client = MongoClient("mongodb://172.22.85.17:27017/")
    db = mongo_client["model_updates"]
    collection = db["weights"]
    logging.info("MongoDB connection successful!")
except Exception as e:
    logging.error(f"Error connecting to MongoDB: {e}")

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

global_model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(global_model.parameters(), lr=0.01)

class WeightsUpdate(BaseModel):
    client_id: str
    model_weights: List[List[float]]
    num_samples: int
    loss: float

class ClientData(BaseModel):
    local_losses: Dict[str, float]
    num_users: int
    num_selected: int

current_version = 0
client_updates = []

@app.get("/get_model")
async def get_model():
    global global_model, current_version
    updated_model_state_dict = global_model.state_dict()
    updated_model_state_dict_json = {key: value.tolist() for key, value in updated_model_state_dict.items()}
    return {"message": "Global model fetched successfully", "model_state_dict": updated_model_state_dict_json, "version": current_version}

@app.post("/update_model")
async def update_model(weights: WeightsUpdate):
    global client_updates, current_version

    client_updates.append(weights)
    logging.info(f"Received model update from client {weights.client_id} with loss {weights.loss}")

    if len(client_updates) < 3:
        return {"message": f"Waiting for more client updates. Received {len(client_updates)}/3 updates"}

    # Select top 2 clients based on loss
    client_updates.sort(key=lambda x: x.loss, reverse=True)
    selected_clients = client_updates[:2]
    remaining_clients = client_updates[2:]

    # Update the global model using the selected clients' weights
    model_weights_list = [client.model_weights for client in selected_clients]
    updated_model_state_dict = fed_avg(global_model, model_weights_list)
    global_model.load_state_dict(updated_model_state_dict)

    client_updates = remaining_clients  # Clear updates for next round

    current_version += 1
    updated_model_state_dict_json = {key: value.tolist() for key, value in updated_model_state_dict.items()}

    task_result = store_aggregated_weights_task.delay(updated_model_state_dict_json)
    inserted_id = task_result.get()

    logging.info(f"Global model updated. Inserted ID: {inserted_id}, Version: {current_version}")

    return {
        "message": "Global model updated",
        "model_state_dict": updated_model_state_dict_json,
        "version": current_version,
        "inserted_id": inserted_id,
        "selected_clients": [client.client_id for client in selected_clients]
    }

@app.post("/send_updated_model")
async def send_updated_model():
    global global_model, current_version
    updated_model_state_dict = global_model.state_dict()
    updated_model_state_dict_json = {key: value.tolist() for key, value in updated_model_state_dict.items()}
    return {
        "message": "Updated global model sent successfully",
        "model_state_dict": updated_model_state_dict_json,
        "version": current_version
    }

@app.post("/select_clients/biggest_loss")
async def select_biggest_loss(data: ClientData):
    sorted_clients = sorted(data.local_losses.items(), key=lambda item: item[1], reverse=True)
    selected_clients = [client for client, _ in sorted_clients[:data.num_selected]]
    return {"selected_clients": selected_clients}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
