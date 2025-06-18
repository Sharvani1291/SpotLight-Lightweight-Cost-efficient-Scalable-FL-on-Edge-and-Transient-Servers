import torch

def fed_prox(global_model, weights_updates):
    """
    FedProx aggregation function.
    
    In FedProx, the proximal term is applied during the client's local update.
    Thus, the aggregation step is identical to simple weighted averaging (like FedAvg).
    
    Args:
        global_model (torch.nn.Module): The current global model.
        weights_updates (list): A list of state_dicts from each client.
        
    Returns:
        dict: The updated global model's state_dict.
    
    Example usage:
        model_weights_list = [client.model_weights for client in client_updates]
        updated_model_state_dict = fed_prox(model, model_weights_list)
    """
    new_state_dict = global_model.state_dict()
    
    for key in new_state_dict.keys():
        aggregated = torch.zeros_like(new_state_dict[key])
        for w in weights_updates:
            aggregated += torch.tensor(w[key], dtype=new_state_dict[key].dtype)
        aggregated.div_(len(weights_updates))
        new_state_dict[key] = aggregated

    global_model.load_state_dict(new_state_dict)
    print("FedProx aggregation is successful!")
    return new_state_dict
