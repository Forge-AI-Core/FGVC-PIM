import torch

def get_device():
    """
    Returns the best available device (CUDA, MPS, or CPU).
    """
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
