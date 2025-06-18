import torch

def fed_nova(global_model, weights_updates, local_steps_updates, lr=0.01):
    """
    Performs FedNova aggregation.

    For each parameter in the global model, compute the normalized update:
        delta = (client_param - global_param) / local_steps
    Then average these deltas across clients and update the global model as:
         new_param = global_param + lr * average_delta

    Args:
        global_model (torch.nn.Module): The current global model.
        weights_updates (list): A list of state_dicts (one per client) with client weights.
        local_steps_updates (list): A list of integers corresponding to the number of local
                                    update steps performed by each client.
        lr (float, optional): Server learning rate. Defaults to 0.01.

    Returns:
        dict: The updated global model's state_dict.
    """
    new_state_dict = global_model.state_dict()
    for key in new_state_dict.keys():
        aggregated_delta = torch.zeros_like(new_state_dict[key])
        for client_state, local_steps in zip(weights_updates, local_steps_updates):
            client_param = torch.tensor(client_state[key], dtype=new_state_dict[key].dtype)
            delta = (client_param - new_state_dict[key]) / local_steps
            aggregated_delta += delta
        aggregated_delta.div_(len(weights_updates))
        new_state_dict[key] = new_state_dict[key] + lr * aggregated_delta

    global_model.load_state_dict(new_state_dict)
    print("FedNova aggregation is successful!")
    return new_state_dict
