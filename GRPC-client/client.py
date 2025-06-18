import grpc
import spotlight_pb2
import spotlight_pb2_grpc
from spotlight_pb2 import ModelRequest, WeightsUpdate
from spotlight_pb2_grpc import ModelServiceStub
import torch
import torch.optim as optim
import logging
from datetime import datetime
from torch.utils.data import Subset, Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import datasets, transforms
from pymongo import MongoClient
from models import get_model
import sys
import os
import torch.nn as nn
import bz2
import base64
import pickle
import numpy as np
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

logging.basicConfig(level=logging.INFO)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == 'cuda':
    logging.info("Using GPU for training.")
else:
    logging.info("Using CPU for training.")

import argparse

parser = argparse.ArgumentParser(description="Train a model on MNIST, CIFAR-10, or CIFAR-100")
parser.add_argument(
    "--model",
    type=str,
    choices=["simplecnn", "twocnn", "logistic_regression", "squeezenet", "mobilenet", "resnet", "densenet"],
    default="simplecnn",
    help="Specify the model name (simplecnn, twocnn, logistic_regression, squeezenet, resnet, densenet)"
)
parser.add_argument(
    "--dataset",
    type=str,
    choices=["mnist", "cifar10", "cifar100", "imagenet"],
    default="mnist",
    help="Specify the dataset (mnist, cifar10, cifar100, or imagenet)"
)

compressed_models = ["resnet", "densenet", "squeezenet", "mobilenet"]
args = parser.parse_args()
model_name = args.model
dataset_name = args.dataset

#Load the selected model dynamically
try:
    selected_model = get_model(model_name, dataset_name).to(device)
except ValueError as e:
    logging.error(f"Model selection error: {e}")
    exit(1)

#Optimizer setup (SGD for CNNs, Adam for others)
if model_name in ["simplecnn", "twocnn", "squeezenet", "resnet", "densenet", "mobilenet"]:
    optimizer = optim.SGD(selected_model.parameters(), lr=0.001, momentum=0.9)
else:
    optimizer = optim.Adam(selected_model.parameters(), lr=0.001)
    
criterion = nn.CrossEntropyLoss()

#Set accuracy thresholds dynamically
if model_name == "logistic_regression":
    accuracy_threshold = 25.0
elif model_name == "squeezenet":
    accuracy_threshold = 10.0
elif model_name in ["resnet", "densenet"]:
    accuracy_threshold = 20.0
elif model_name in "mobilenet":
    accuracy_threshold = 80.0
else:
    accuracy_threshold = 75.0

current_version = 0

def fetch_global_model(model):
    """Fetches the global model from the server via gRPC."""
    global current_version, model_name
    logging.info("Attempting to fetch the global model.")
    
    server_address = "localhost:50051"
    #Increase the max message size
    options = [('grpc.max_receive_message_length', 100 * 1024 * 1024)]
    
    try:
        channel = grpc.insecure_channel(server_address, options=options)
        stub = ModelServiceStub(channel)
        response = stub.GetModel(spotlight_pb2.ModelRequest(model_name=model_name))

        global_model_weights = {key: torch.tensor(value) for key, value in response.model_state_dict.items()}
        model.load_state_dict(global_model_weights, strict=False)
        current_version = response.version

        logging.info(f"Fetched global model version {current_version}.")
        return True
    except grpc.RpcError as e:
        logging.error(f"gRPC error while fetching global model: {e}")
        return False
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return False

def send_local_model_weights(weights, version, loss, total_training_time, accuracy):
    """Sends local model weights to the server with compression for large models."""
    logging.info("Preparing to send local model weights to the server.")

    is_compressed = False
    model_weights_proto = {}

    try:
        if model_name in compressed_models:
            logging.info(f"Compressing weights for {model_name} before sending.")

            # Convert weights to NumPy float16 for efficient storage
            model_weights_np = {key: value.cpu().detach().numpy().astype(np.float16) for key, value in weights.items()}

            # Serialize and compress using BZ2 + Base64
            compressed_weights = bz2.compress(pickle.dumps(model_weights_np))

            #Use `compressed_weights` field in protobuf
            model_weights_proto["compressed"] = spotlight_pb2.ModelWeights(compressed_weights=compressed_weights)

            is_compressed = True  #Mark as compressed

        else:
            #Use standard float16 for smaller models (no compression)
            for key, value in weights.items():
                try:
                    weights_as_np = value.cpu().detach().numpy().astype(np.float16)
                    model_weights_proto[key] = spotlight_pb2.ModelWeights(values=weights_as_np.flatten().tolist())
                except Exception as e:
                    logging.error(f"Error processing weights for key {key}: {e}")
                    continue

            is_compressed = False  #No compression for smaller models

        #Prepare weight update request
        weight_update = spotlight_pb2.WeightsUpdate(
            client_id="1",
            model_name=model_name,
            model_weights=model_weights_proto,
            num_samples=len(weights),
            loss=loss,
            version=version,
            Client_training_time=total_training_time,
            compressed=is_compressed  #Inform server if compression was used
        )

        #Increase max send size to 100MB
        server_address = "localhost:50051"
        options = [
            ('grpc.max_send_message_length', 100 * 1024 * 1024),  #Increase to 100MB
            ('grpc.max_receive_message_length', 100 * 1024 * 1024)  #Increase to 100MB
        ]

        channel = grpc.insecure_channel(server_address, options=options)
        stub = spotlight_pb2_grpc.ModelServiceStub(channel)
        response = stub.UpdateModel(weight_update)

        logging.info(f"Model weights sent successfully. Server response: {response.message}")
        return True

    except grpc.RpcError as e:
        logging.error(f"gRPC error while sending model weights: {e}")
        return False
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return False

def train_local_model(train_loader, model, optimizer, criterion, device):
    """Trains the local model."""
    model.train()
    for epoch in range(10): 
        running_loss = 0.0
        logging.info(f"Training Epoch {epoch + 1}")
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            #Ensure target labels are within range [0, n_classes-1]
            num_classes = 100 if dataset_name == "cifar100" else 10
            target = torch.clamp(target, 0, num_classes - 1)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if batch_idx % 100 == 0:
                logging.info(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item()}")
        avg_loss = running_loss / len(train_loader)
        logging.info(f"Epoch {epoch + 1}, Average Loss: {avg_loss}")
        
    return avg_loss, model.state_dict()

def calculate_accuracy(model, data_loader, device):
    """Calculates model accuracy."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total
    logging.info(f"Accuracy: {accuracy:.2f}%")
    return accuracy

def start_training(train_loader, val_loader):
    """Manages the entire training process."""
    logging.info("Starting the training process.")

    while True:
        # Fetch global model
        if not fetch_global_model(selected_model):
            logging.error("Failed to fetch global model. Retrying...")
            continue

        # Train local model
        loss_value, trained_weights = train_local_model(train_loader, selected_model, optimizer, nn.CrossEntropyLoss(), device)
        logging.info(f"Completed local training. Loss: {loss_value:.4f}")

        # Calculate accuracy
        accuracy = calculate_accuracy(selected_model, val_loader, device)
        logging.info(f"Validation accuracy: {accuracy:.2f}%")

        # Send model weights if accuracy is sufficient
        if accuracy >= accuracy_threshold:
            logging.info(f"Accuracy threshold of {accuracy_threshold}% reached. Sending weights to server.")
            send_local_model_weights(trained_weights, current_version, loss_value, 0, accuracy)
            break

def get_mnist_data():
    """Loads the MNIST dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST('mnist_data', train=True, transform=transform, download=True)
    val_dataset = datasets.MNIST('mnist_data', train=False, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)
    return train_loader, val_loader

def get_cifar10_data():
    """Loads the CIFAR-10 dataset."""
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    train_dataset = datasets.CIFAR10('cifar10_data', train=True, transform=transform, download=True)
    val_dataset = datasets.CIFAR10('cifar10_data', train=False, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)
    return train_loader, val_loader

def get_cifar100_data():
    """Loads the CIFAR-100 dataset (for ResNet and DenseNet)."""
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
    ])
    train_dataset = datasets.CIFAR100('cifar100_data', train=True, transform=transform, download=True)
    val_dataset = datasets.CIFAR100('cifar100_data', train=False, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
    return train_loader, val_loader

class SingleImageDataset(Dataset):
    def __init__(self, image_path, transform=None):
        self.image_path = image_path
        self.transform = transform
        self.image = Image.open(self.image_path).convert("RGB")

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if self.transform:
            img = self.transform(self.image)
        else:
            img = self.image
        label = 0
        return img, label

def get_imagenet_data(root_dir="/home/cloudlabgpu1/grpc_spotlight/GRPC-client/imagenet_data"):
    """Loads a single image from ImageNet for fine-tuning or testing."""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])
    train_dataset = ImageFolder(root=root_dir, transform=transform)
    val_dataset = ImageFolder(root=root_dir, transform=transform)
    
    logging.info(f"Total training images loaded: {len(train_dataset)}")
    logging.info(f"Total validation images loaded: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    return train_loader, val_loader


if __name__ == "__main__":
    logging.info(f"Using dataset: {dataset_name}")

    #Load the chosen dataset dynamically
    if dataset_name == "mnist":
        train_loader, val_loader = get_mnist_data()
    elif dataset_name == "cifar10":
        train_loader, val_loader = get_cifar10_data()
    elif dataset_name == "imagenet":
        train_loader, val_loader = get_imagenet_data("/home/cloudlabgpu1/grpc_spotlight/GRPC-client/imagenet_data")
    elif dataset_name == "cifar100":
        train_loader, val_loader = get_cifar100_data()

    start_training(train_loader, val_loader)
