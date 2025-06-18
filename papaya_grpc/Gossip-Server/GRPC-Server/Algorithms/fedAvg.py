# Algorithms/fedAvg.py
import torch

def apply_fedavg(global_model, client_weights, client_sample_counts):
    """
    Applies the FedAvg algorithm to aggregate client weights.

    Args:
        global_model (nn.Module): The global model to update.
        client_weights (list): List of tensors representing client model weights.
        client_sample_counts (list): List of sample counts for each client.

    Returns:
        None: The global model is updated in-place.
    """
    total_samples = sum(client_sample_counts)
    avg_weights = torch.zeros_like(client_weights[0])

    for weight, num_samples in zip(client_weights, client_sample_counts):
        avg_weights += weight * (num_samples / total_samples)

    offset = 0
    for param in global_model.parameters():
        param_size = param.numel()
        param.data.copy_(avg_weights[offset:offset + param_size].view_as(param))
        offset += param_size
