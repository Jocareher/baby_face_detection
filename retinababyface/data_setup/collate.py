from typing import List, Dict, Any

import torch
import torch.nn.functional as F


def custom_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function that stacks images, pads the target tensors,
    and generates a 'valid_mask' to indicate the presence of valid objects.

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
    valid_masks = []  # List to store valid object masks.

    for item in batch:  # Iterate through each sample in the batch.
        boxes = item["target"]["boxes"]  # Get the bounding box tensor.
        angles = item["target"]["angles"]  # Get the angle tensor.
        classes = item["target"]["class_idx"]  # Get the class index tensor.
        n = boxes.shape[0]  # Get the number of objects in the current sample.
        pad_size = max_num_objs - n  # Calculate the padding size.

        # Handle the case where there are no objects (pure background).
        if n == 0:
            boxes = torch.zeros((0, 8), dtype=torch.float32)
            angles = torch.zeros((0,), dtype=torch.float32)
            classes = torch.full((0,), fill_value=5, dtype=torch.long)  # 5 = no face

        padded_boxes.append(
            F.pad(boxes, (0, 0, 0, pad_size))
        )  # Pad the bounding boxes tensor.
        padded_angles.append(F.pad(angles, (0, pad_size)))  # Pad the angles tensor.
        padded_classes.append(
            F.pad(classes, (0, pad_size), value=5)
        )  # Pad the class indices tensor.

        mask = torch.zeros(max_num_objs, dtype=torch.bool)  # Create a mask tensor.
        mask[:n] = True  # Set the first 'n' elements to True.
        valid_masks.append(mask)  # Append the mask to the list.

    targets = {
        "boxes": torch.stack(padded_boxes),  # Stack the padded bounding boxes.
        "angles": torch.stack(padded_angles),  # Stack the padded angles.
        "class_idx": torch.stack(padded_classes),  # Stack the padded class indices.
        "valid_mask": torch.stack(valid_masks),  # Stack the valid object masks.
    }

    return {
        "image": images,
        "target": targets,
    }  # Return the stacked images and padded targets.
