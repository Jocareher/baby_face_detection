import os
import random
from pathlib import Path

import numpy as np
import torch

from typing import Any, Optional, Dict


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
    Otherwise, it returns "cpu".

    Returns:
        torch.device: The default PyTorch device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")  # Use CUDA if available.
    # elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    #     return torch.device("mps")  # Use MPS if available.
    else:
        return torch.device("cpu")  # Use CPU if no GPU is available.


def seed_worker(worker_id: int) -> None:
    """
    Seed function for DataLoader workers to ensure reproducibility.
    Args:
        worker_id (int): The worker ID provided by DataLoader.
    Returns:
        None
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def to_numpy(x: Any) -> Optional[np.ndarray]:
    """
    Converts the input to a NumPy array.

    Handles PyTorch Tensors by detaching them from the computational graph
    and moving them to the CPU before conversion.

    Args:
        x: The input object, which can be None, a PyTorch Tensor, or any
           object convertible by np.asarray.

    Returns:
        The resulting NumPy ndarray, or None if the input was None.
    """
    if x is None:
        # Return None directly if the input is None
        return None
    if isinstance(x, torch.Tensor):
        # Detach from graph, move to CPU, and convert to NumPy
        return x.detach().cpu().numpy()
    # Convert other types (lists, tuples, existing NumPy arrays, etc.)
    return np.asarray(x)


def ensure_polygons_42_shape(polys_np: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """
    Standardizes a batch of polygons to the (N, 4, 2) float32 format.

    Accepts polygons in two shapes:
    1. Flat format: (N, 8) where N is the number of polygons.
    2. Per-vertex format: (N, 4, 2) where N is the number of polygons.

    Args:
        polys_np: A NumPy array of polygons (N, 8) or (N, 4, 2), or None.

    Returns:
        A NumPy array of polygons with shape (N, 4, 2) and dtype float32,
        or None if the input was None or empty.

    Raises:
        ValueError: If the input array has a shape that is not supported.
    """
    if polys_np is None:
        # Handle None input
        return None

    # Ensure input is a standard NumPy array, handling potential PyTorch Tensors
    polys_np = to_numpy(polys_np)

    if polys_np.size == 0:
        # Handle empty array
        return None

    if polys_np.ndim == 2 and polys_np.shape[1] == 8:
        # (N, 8) format: reshape to (N, 4, 2)
        return polys_np.reshape(-1, 4, 2).astype(np.float32)

    if polys_np.ndim == 3 and polys_np.shape[1:] == (4, 2):
        # (N, 4, 2) format: ensure correct dtype
        return polys_np.astype(np.float32)

    # Raise an error for unsupported shapes
    raise ValueError(f"Unsupported polygon shape: {polys_np.shape}")

def resolve_image_path(
    batch: Dict[str, Any],
    b: int,
    global_idx: int,
    dataset: Any = None,
) -> Path:
    """
    Try to resolve the path for the b-th sample of the current batch.
    Priority:
      1) batch["paths"][b]               (si el collate lo incluye)
      2) dataset.paths[global_idx]       (o file_list/files/imgs/images/samples/items)
      3) fallback: sample_XXXXXX.jpg
    """
    # 1) Del batch (si tu images_only_collate incluye "paths")
    if isinstance(batch, dict) and "paths" in batch:
        paths = batch["paths"]
        if isinstance(paths, (list, tuple)) and len(paths) > b:
            return Path(paths[b])

    # 2) Del dataset (lista de atributos comunes)
    if dataset is not None:
        for attr in ("paths", "file_list", "files", "imgs", "images", "samples", "items"):
            if hasattr(dataset, attr):
                obj = getattr(dataset, attr)
                try:
                    if isinstance(obj, (list, tuple)) and len(obj) > global_idx:
                        item = obj[global_idx]
                        # Si es (path, label) u otra tupla, tomar el path en [0]s
                        if isinstance(item, (list, tuple)) and item and isinstance(item[0], (str, Path)):
                            return Path(item[0])
                        if isinstance(item, (str, Path)):
                            return Path(item)
                except Exception:
                    pass

    # 3) Respaldo
    return Path(f"sample_{global_idx:06d}.jpg")
