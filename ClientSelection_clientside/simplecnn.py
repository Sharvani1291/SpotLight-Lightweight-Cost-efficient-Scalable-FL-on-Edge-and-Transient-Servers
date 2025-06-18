import requests
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Subset
from pydantic import BaseModel
from typing import List
import logging

logging.basicConfig(level=logging.INFO)

client_id = "edg2_jetson" 

class WeightUpdate(BaseModel):
    client_id: str
    model_weights: List[List[float]]
    num_samples: int
    loss: float

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

criterion = nn.CrossEntropyLoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
simple_cnn_model = SimpleCNN().to(device)
optimizer = optim.SGD(simple_cnn_model.parameters(), lr=0.001)

current_version = 0

def train_local_model(train_loader, model, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    num_batches = len(train_loader)
    for epoch in range(2):
        running_loss = 0.0
        logging.info(f"Training Epoch {epoch + 1}")
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if batch_idx % 100 == 0:
                logging.info(f"Batch {batch_idx}/{num_batches}, Loss: {loss.item()}")
        logging.info(f"Epoch {epoch + 1}, Average Loss: {running_loss / num_batches}")
        total_loss += running_loss

    average_loss = total_loss / (num_batches * 2)  # Assuming 2 epochs
    return model.state_dict(), average_loss

def send_local_model_weights(weights, version, loss):
    model_weights_list = [value.flatten().tolist() for value in weights.values()]
    weight_update = WeightUpdate(client_id=client_id, model_weights=model_weights_list, num_samples=len(weights), loss=loss)

    response = requests.post("http://172.22.86.230:8000/update_model", json=weight_update.dict())

    if response.status_code == 200:
        response_data = response.json()
        if "selected_clients" in response_data:
            if client_id in response_data["selected_clients"]:
                logging.info("Selected for next round of training")
                return True
            else:
                logging.info("Not selected for next round of training")
                return False
        logging.info("Model update successful")
        return True
    elif response.status_code == 409:
        logging.warning("Model version mismatch. Fetching the latest model.")
        fetch_global_model(simple_cnn_model)
        send_local_model_weights(weights, version, loss)
    else:
        logging.error(f"Failed to send model update: {response.status_code}")
        return False

def fetch_global_model(model):
    global current_version
    response = requests.get("http://172.22.86.230:8000/get_model")
    if response.status_code == 200:
        data = response.json()
        global_model_weights = data["model_state_dict"]
        current_version = data["version"]
        
        global_model_weights = {key: torch.tensor(value) for key, value in global_model_weights.items()}
        model.load_state_dict(global_model_weights, strict=False)
        logging.info(f"Global model updated to version {current_version}")
    else:
        logging.error("Failed to fetch global model")

def fetch_updated_model():
    response = requests.post("http://172.22.86.230:8000/send_updated_model")
    if response.status_code == 200:
        data = response.json()
        global_model_weights = data["model_state_dict"]
        current_version = data["version"]
        
        global_model_weights = {key: torch.tensor(value) for key, value in global_model_weights.items()}
        return global_model_weights, current_version
    else:
        logging.error("Failed to fetch updated model")
        return None, None

def update_local_model(global_model_weights):
    global simple_cnn_model
    simple_cnn_model.load_state_dict(global_model_weights, strict=False)

def start_training(train_loader):
    global simple_cnn_model, optimizer, criterion, device
    
    for epoch in range(2):
        running_loss = 0.0
        simple_cnn_model.train()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = simple_cnn_model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            if batch_idx % 100 == 0:
                logging.info(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item()}")
        
        logging.info(f"Epoch {epoch + 1}, Average Loss: {running_loss / len(train_loader)}")
    
    return simple_cnn_model.state_dict(), running_loss / len(train_loader)

def get_train_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('mnist_data', train=True, transform=transform, download=True)
    subset_train_dataset = Subset(train_dataset, indices=range(10000))
    train_loader = torch.utils.data.DataLoader(subset_train_dataset, batch_size=64, shuffle=True)
    
    return train_loader

if __name__ == "__main__":
    fetch_global_model(simple_cnn_model)
    train_loader = get_train_data()
    trained_weights, average_loss = start_training(train_loader)
    if send_local_model_weights(trained_weights, current_version, average_loss):
        updated_model_weights, updated_version = fetch_updated_model()
        if updated_model_weights:
            simple_cnn_model.load_state_dict(updated_model_weights, strict=False)
            train_loader = get_train_data()
            start_training(train_loader)
