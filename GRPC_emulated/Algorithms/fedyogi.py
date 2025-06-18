import torch

_yogi_m = None  # First moment estimates
_yogi_v = None  # Second moment estimates
_yogi_t = 0     # Round counter

# Hyperparameters 
_yogi_lr = 0.01
_yogi_beta1 = 0.9
_yogi_beta2 = 0.99
_yogi_eps = 1e-8

def fed_yogi(global_model, weights_updates):
    """
    FedYogi aggregation:
      - Compute the average client model state (like FedAvg).
      - Compute delta = (avg_client - global_state) for each parameter.
      - Update first moment (_yogi_m) as in Adam.
      - Update second moment (_yogi_v) using the Yogi rule:
             _yogi_v = _yogi_v - (1 - beta2) * (delta**2) * sign(_yogi_v - delta**2)
      - Apply bias correction and update the global model.
    
    Args:
      global_model (torch.nn.Module): The current global model.
      weights_updates (list): A list of state_dicts from clients.
    
    Returns:
      dict: The updated global model's state_dict.
    
    Example usage in your UpdateModel handler:
    
        model_weights_list = [client.model_weights for client in client_updates]
        updated_model_state_dict = fed_yogi(model, model_weights_list)
    """
    global _yogi_m, _yogi_v, _yogi_t, _yogi_lr, _yogi_beta1, _yogi_beta2, _yogi_eps

    global_state = global_model.state_dict()

    if _yogi_m is None or _yogi_v is None:
        _yogi_m = {key: torch.zeros_like(val) for key, val in global_state.items()}
        _yogi_v = {key: torch.zeros_like(val) for key, val in global_state.items()}

    avg_state = {}
    for key in global_state.keys():
        avg_state[key] = torch.zeros_like(global_state[key])
        for w in weights_updates:
            avg_state[key] += torch.tensor(w[key], dtype=global_state[key].dtype)
        avg_state[key].div_(len(weights_updates))

    _yogi_t += 1  
    for key in global_state.keys():
        delta = avg_state[key] - global_state[key]
        # Update first moment: 
        # m = beta1 * m + (1-beta1)*delta
        _yogi_m[key] = _yogi_beta1 * _yogi_m[key] + (1 - _yogi_beta1) * delta
        # Update second moment using Yogi rule:
        # v = v - (1-beta2) * (delta**2) * sign(v - delta**2)
        _yogi_v[key] = _yogi_v[key] - (1 - _yogi_beta2) * (delta ** 2) * torch.sign(_yogi_v[key] - (delta ** 2))
        
        # Bias corrections:
        m_hat = _yogi_m[key] / (1 - _yogi_beta1 ** _yogi_t)
        v_hat = _yogi_v[key] / (1 - _yogi_beta2 ** _yogi_t)
        
        global_state[key] = global_state[key] - _yogi_lr * m_hat / (torch.sqrt(v_hat) + _yogi_eps)

    global_model.load_state_dict(global_state)
    print("FedYogi aggregation is successful!")
    return global_state
