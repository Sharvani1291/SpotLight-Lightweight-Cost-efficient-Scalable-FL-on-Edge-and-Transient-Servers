import torch

def fed_avg(global_model, weights_updates):
    new_state_dict = global_model.state_dict()

    for key, value in new_state_dict.items():
        key_suffix = key.split('.')[-1]

        for w in weights_updates:
            if key_suffix in w:
                value += torch.tensor(w[key_suffix]) / len(weights_updates)

    global_model.load_state_dict(new_state_dict)
    print("Fed average is successful!")
    return global_model.state_dict()
