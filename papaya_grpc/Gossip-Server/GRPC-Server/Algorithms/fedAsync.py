# Algorithms/fedAsync.py
import torch
import math

def apply_fedasync(global_model, client_updates):
    """
    Applies the FedAsync algorithm to aggregate client weights considering staleness.

    Args:
        global_model (nn.Module): The global model to update.
        client_updates (list): List of tuples containing client weights, num_samples, and staleness.

    Returns:
        None: The global model is updated in-place.
    """
    decay_rate = 0.1  # Exponential decay factor for staleness
    total_weight = 0.0
    weighted_avg = torch.zeros_like(client_updates[0][0])

    for client_weights, num_samples, staleness in client_updates:
        staleness_factor = math.exp(-decay_rate * staleness)
        weight = num_samples * staleness_factor
        weighted_avg += client_weights * weight
        total_weight += weight

    # Normalize weights
    if total_weight > 0:
        weighted_avg /= total_weight

    # Update the global model
    offset = 0
    for param in global_model.parameters():
        param_size = param.numel()
        param.data.copy_(weighted_avg[offset:offset + param_size].view_as(param))
        offset += param_size
