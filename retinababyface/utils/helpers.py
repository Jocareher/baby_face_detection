import os
import random
import numpy as np
import torch


def set_seed(seed_value: int = 42) -> None:
    """
    Set the random seed for various libraries to ensure reproducibility.

    Args:
        seed_value (int): The random seed value to be used.

    Returns:
        None
    """
    # Set the seed for Python's built-in random library
    os.environ["PYTHONHASHSEED"] = str(seed_value)

    # Setting the random seed for numpy's random number generator
    np.random.seed(seed_value)

    # Setting the random seed for PyTorch's random number generator and the CUDA random number generator
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

    # Setting the random seed for Python's built-in random library
    random.seed(seed_value)

    # Setting the random seed for PyTorch's random number generator
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_default_device() -> torch.device:
    """
    Determines the default device to use for PyTorch computations.

    If a CUDA-enabled GPU is available, it returns "cuda".
    If an MPS-enabled GPU is available, it returns "mps".
    Otherwise, it returns "cpu".

    Returns:
        torch.device: The default PyTorch device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")  # Use CUDA if available.
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")  # Use MPS if available.
    else:
        return torch.device("cpu")  # Use CPU if no GPU is available.
