from typing import List, Dict, Any

import torch
import torch.nn.functional as F


def custom_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate a batch of samples, stacking images and padding per-sample object tensors.
    Uses `valid_mask` to mark which entries are real, so no explicit 'background' class is needed.

    Args:
        batch (List[Dict[str, Any]]): A list of dictionaries, where each dictionary represents a sample
                                      with "image" and "target" keys.

    Returns:
        Dict[str, Any]: A dictionary containing the stacked images and padded targets, including a valid mask.
            - "image": A tensor of stacked images (B, C, H, W).
            - "target": A dictionary containing:
                - "boxes": A tensor of padded bounding boxes (B, max_N, 8).
                - "angles": A tensor of padded angles (B, max_N).
                - "class_idx": A tensor of padded class indices (B, max_N).
                - "child_prob": A tensor of padded child probabilities (B, max_N).
                - "valid_mask": A boolean mask indicating valid object positions (B, max_N).
    """
    images = torch.stack(
        [item["image"] for item in batch], dim=0
    )  # Stack images into a single tensor.

    # Determine the maximum number of objects per image in the batch.
    max_num_objs = max(item["target"]["boxes"].shape[0] for item in batch)

    padded_boxes = []  # List to store padded bounding box tensors.
    padded_angles = []  # List to store padded angle tensors.
    padded_classes = []  # List to store padded class index tensors.
    padded_child_probs = []
    valid_masks = []  # List to store valid object masks.

    for item in batch:  # Iterate through each sample in the batch.
        boxes = item["target"]["boxes"]  # Get the bounding box tensor.
        angles = item["target"]["angles"]  # Get the angle tensor.
        classes = item["target"]["class_idx"]  # Get the class index tensor.
        child_p = item["target"]["child_prob"]  # Get the child probability tensor.
        n = boxes.shape[0]  # Get the number of objects in the current sample.
        pad_size = max_num_objs - n  # Calculate the padding size.

        # Handle the case where there are no objects (pure background).
        if n == 0:
            boxes = torch.zeros((0, 8), dtype=torch.float32)
            angles = torch.zeros((0,), dtype=torch.float32)
            classes = torch.zeros((0,), dtype=torch.long)
            child_p = torch.zeros((0,), dtype=torch.float32)

        padded_boxes.append(
            F.pad(boxes, (0, 0, 0, pad_size))
        )  # Pad the bounding boxes tensor.
        padded_angles.append(F.pad(angles, (0, pad_size)))  # Pad the angles tensor.
        padded_classes.append(
            F.pad(classes, (0, pad_size), value=-1)
        )  # Pad the class indices tensor.
        padded_child_probs.append(  # Pad the child probabilities tensor.
            F.pad(child_p, (0, pad_size), value=-1.0)
        )

        mask = torch.zeros(max_num_objs, dtype=torch.bool)  # Create a mask tensor.
        mask[:n] = True  # Set the first 'n' elements to True.
        valid_masks.append(mask)  # Append the mask to the list.

    targets = {
        "boxes": torch.stack(padded_boxes),  # Stack the padded bounding boxes.
        "angles": torch.stack(padded_angles),  # Stack the padded angles.
        "class_idx": torch.stack(padded_classes),  # Stack the padded class indices.
        "child_prob": torch.stack(
            padded_child_probs
        ),  # Stack the padded child probabilities.
        "valid_mask": torch.stack(valid_masks),  # Stack the valid object masks.
    }

    return {
        "image": images,
        "target": targets,
    }  # Return the stacked images and padded targets.


def images_only_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate for images-only datasets. Returns only 'image' and empty 'target'.s
    """
    # Filter out None (if your dataset returns None on read errors)
    batch = [b for b in batch if b is not None]
    if not batch:
        return {
            "image": torch.empty(0),
            "target": {
                "boxes": torch.empty((0, 0, 8), dtype=torch.float32),
                "angles": torch.empty((0, 0), dtype=torch.float32),
                "class_idx": torch.empty((0, 0), dtype=torch.long),
                "child_prob": torch.empty((0, 0), dtype=torch.float32),
                "valid_mask": torch.empty((0, 0), dtype=torch.bool),
            },
            # no 'paths', no 'meta'
        }

    imgs = [b["image"] for b in batch]
    imgs = [im if isinstance(im, torch.Tensor) else torch.as_tensor(im) for im in imgs]
    # If HWC -> CHW
    imgs = [
        im.permute(2, 0, 1)
        if im.ndim == 3 and im.shape[-1] in (3, 4) and im.shape[0] not in (1, 3, 4)
        else im
        for im in imgs
    ]
    images = torch.stack(imgs, dim=0)

    B = len(batch)
    targets = {
        "boxes": torch.zeros((B, 0, 8), dtype=torch.float32),
        "angles": torch.zeros((B, 0), dtype=torch.float32),
        "class_idx": torch.zeros((B, 0), dtype=torch.long),
        "child_prob": torch.zeros((B, 0), dtype=torch.float32),
        "valid_mask": torch.zeros((B, 0), dtype=torch.bool),
    }

    return {"image": images, "target": targets}
