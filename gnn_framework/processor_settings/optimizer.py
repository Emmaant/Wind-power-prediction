import torch.optim as optim

def get_optimizer(optimizer_name, params, lr, momentum=0.9):

    if not params:
        raise ValueError("Optimizer got an empty parameter list inside function!")

    if optimizer_name == "SGD":
        return optim.SGD(params, lr=lr, momentum=momentum)
    elif optimizer_name == "Adam":
        return optim.Adam(params, lr=lr)
    else:
        raise ValueError(f"Invalid optimizer name: {optimizer_name}. Choose 'SGD' or 'Adam'.")





