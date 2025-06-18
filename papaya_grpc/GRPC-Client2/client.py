import grpc
import spotlight_pb2
import spotlight_pb2_grpc
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms
import logging
import uuid
import csv
from datetime import datetime
import os
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the CNN model
# class SimpleCNN(nn.Module):
#     def __init__(self):
#         super(SimpleCNN, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.fc1 = nn.Linear(64 * 7 * 7, 128)
#         self.fc2 = nn.Linear(128, 10)

#     def forward(self, x):
#         x = self.pool(nn.functional.relu(self.conv1(x)))
#         x = self.pool(nn.functional.relu(self.conv2(x)))
#         x = x.view(-1, 64 * 7 * 7)
#         x = nn.functional.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x


class SingleLayerNN(nn.Module):
    def __init__(self):
        super(SingleLayerNN, self).__init__()
        self.fc = nn.Linear(784, 10)  # MNIST images are 28x28 pixels

    def forward(self, x):
        x = x.view(-1, 784)  # Flatten the image
        x = self.fc(x)
        return x
# Evaluate model accuracy
def evaluate_model(model, device, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy = 100 * correct / total
    return accuracy

# Run the client
def run_client():
    request_count = 0
    client_uuid = str(uuid.uuid4())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    try:
        with grpc.insecure_channel('172.22.86.230:8082') as channel:
            stub = spotlight_pb2_grpc.FederatedLearningServiceStub(channel)

            # Receive the global model weights and version
            global_model_response = stub.GetGlobalModel(spotlight_pb2.ModelVersion(version="v1.0"))
            global_weights = np.array(global_model_response.weights, dtype=np.float32)

            # Initialize the current model version from the server response
            current_model_version = int(global_model_response.version)

            # Initialize the model
            model = SingleLayerNN().to(device)

            # Load global weights into the model
            model_state_dict = model.state_dict()
            offset = 0
            for name, param in model_state_dict.items():
                param_size = param.numel()
                reshaped_weights = global_weights[offset:offset + param_size].reshape(param.shape)
                if reshaped_weights.shape == param.shape:
                    param.data.copy_(torch.tensor(reshaped_weights).to(device))
                else:
                    logging.error(f"Shape mismatch for {name}: expected {param.shape}, got {reshaped_weights.shape}")
                offset += param_size

            # Verify if weights fully matched
            logging.info(f"Total parameters in client model: {sum(p.numel() for p in model.parameters())}")
            logging.info(f"Total weights received from server: {len(global_weights)}")

            # Load training data
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
            train_dataset = datasets.MNIST('.', train=True, download=True, transform=transform)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

            # Define optimizer and loss function
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

            # Train the model
            model.train()
            for epoch in range(2):
                running_loss = 0.0
                for i, (inputs, labels) in enumerate(train_loader):
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                logging.info(f"Epoch [{epoch + 1}/2], Loss: {running_loss:.4f}")

            # Collect trained weights
            client_weights = []
            for param in model.parameters():
                client_weights.extend(param.data.cpu().numpy().flatten())

            # Verify collected weights
            logging.info(f"Total weights collected after training: {len(client_weights)}")

            num_samples = len(train_dataset)
            client_timestamp = time.time()

            # Send weights to the server
            response = stub.SendModelWeights(
                spotlight_pb2.ModelWeights(weights=client_weights, num_samples=num_samples, timestamp=client_timestamp,
                                            version=current_model_version)
            )
            logging.info(f"Server response: {response.message}")

            # Log request
            file_exists = os.path.exists("request_log.csv")
            with open("request_log.csv", mode='a', newline='') as file:
                writer = csv.writer(file)
                if not file_exists:
                    writer.writerow(["Time", "Request Count", "Client UUID", "Global Version"])
                writer.writerow([datetime.now(), request_count + 1, client_uuid, current_model_version])

            # Evaluate model accuracy
            test_dataset = datasets.MNIST('.', train=False, download=True, transform=transform)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)
            accuracy = evaluate_model(model, device, test_loader)
            logging.info(f"Model accuracy: {accuracy:.2f}%")

    except grpc.RpcError as e:
        logging.error(f"gRPC error: {e.code()} - {e.details()}")

    except Exception as e:
        logging.error(f"Unexpected error: {e}")


if __name__ == "__main__":
    run_client()
