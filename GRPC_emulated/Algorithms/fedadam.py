import torch

# Global (module-level) state for FedAdam
_m = None  # First moment estimates
_v = None  # Second moment estimates
_t = 0     # Round counter

# Hyperparameters
_lr = 0.01
_beta1 = 0.9
_beta2 = 0.999
_eps = 1e-8

def fed_adam(global_model, weights_updates):
    """
    FedAdam aggregation using module-level variables for state.
    This function assumes weights_updates is a list of state_dicts (one per client).
    It computes the average client state (like FedAvg), then computes the update delta,
    updates the Adam moments, applies bias correction, and finally updates the global model.
    
    model_weights_list = [client.model_weights for client in client_updates]
    updated_model_state_dict = fed_adam(model, model_weights_list)
    """
    global _m, _v, _t, _lr, _beta1, _beta2, _eps

    global_state = global_model.state_dict()

    if _m is None or _v is None:
        _m = {key: torch.zeros_like(val) for key, val in global_state.items()}
        _v = {key: torch.zeros_like(val) for key, val in global_state.items()}
    
    avg_state = {}
    for key in global_state.keys():
        avg_state[key] = torch.zeros_like(global_state[key])
        for w in weights_updates:
            avg_state[key] += torch.tensor(w[key], dtype=global_state[key].dtype)
        avg_state[key].div_(len(weights_updates))
    
    _t += 1 
    for key in global_state.keys():
        delta = avg_state[key] - global_state[key]
        _m[key] = _beta1 * _m[key] + (1 - _beta1) * delta
        _v[key] = _beta2 * _v[key] + (1 - _beta2) * (delta ** 2)
        m_hat = _m[key] / (1 - _beta1 ** _t)
        v_hat = _v[key] / (1 - _beta2 ** _t)
        global_state[key] = global_state[key] - _lr * m_hat / (torch.sqrt(v_hat) + _eps)
    
    global_model.load_state_dict(global_state)
    print("FedAdam aggregation is successful!")
    return global_state
