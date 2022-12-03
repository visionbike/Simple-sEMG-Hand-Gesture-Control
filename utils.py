import torch


def load_model(model, path):
    # load the model and get the model parameters by using load_state_dict
    model.load_state_dict(torch.load(path))
    return model

