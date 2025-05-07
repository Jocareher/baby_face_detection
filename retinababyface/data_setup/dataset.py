import os
import cv2
from typing import List, Optional, Callable, Dict, Any, Tuple

import torch
from torch.utils.data import Dataset

from .augmentations import Resize


class BabyFacesDataset(Dataset):
    """
    PyTorch Dataset class for loading baby face images and their associated oriented bounding box (OBB) annotations.

    Each image may have one or more annotations stored in a corresponding .txt label file.
    Label format per line:
        class_idx x1 y1 x2 y2 x3 y3 x4 y4 angle

    - class_idx: integer from 0 to 4 indicating face orientation
        (0 = 3/4 leftside, 1 = 3/4 rightside, 2 = frontal, 3 = left profile, 4 = right profile)
    - x1, y1, ..., x4, y4: normalized (0–1) coordinates of the OBB corners
    - angle: rotation angle in radians (clockwise), usually measured from the top-left corner

    Images without a corresponding label file are treated as background and assigned class index 5.

    It is assumed the dataset is organized as:
        root_dir/
            train/
                images/
                labels/
            val/
                images/
                labels/
            ...

    Args:
        root_dir (str): Path to the root directory of the dataset.
        split (str): Subdirectory name indicating the split ("train", "val", "test", etc.). Defaults to "train".
        file_list (Optional[List[str]]): List of image base names (without extension) to load. If None, all .jpg files in the image directory will be used. Defaults to None.
        transform (Optional[Callable]): A function or transform to apply to each sample. Defaults to None.
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        file_list: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
    ):
        """
        Initializes the BabyFacesDataset.

        Args:
            root_dir (str): Path to the root directory of the dataset.
            split (str): Subdirectory name indicating the split ("train", "val", "test", etc.). Defaults to "train".
            file_list (Optional[List[str]]): List of image base names (without extension) to load. If None, all .jpg files in the image directory will be used. Defaults to None.
            transform (Optional[Callable]): A function or transform to apply to each sample. Defaults to None.
        """
        self.root_dir = root_dir  # Assigns the root directory of the dataset.
        self.split = split  # Assigns the split (train, val, test, etc.).
        self.transform = transform  # Assigns the transform to apply to each sample.

        self.images_dir = os.path.join(
            root_dir, split, "images"
        )  # Constructs the path to the images directory.
        self.labels_dir = os.path.join(
            root_dir, split, "labels"
        )  # Constructs the path to the labels directory.

        if file_list is None:  # Checks if a file list is provided.
            self.file_list = (
                [  # Creates a list of image base names from the images directory.
                    os.path.splitext(f)[0]
                    for f in os.listdir(self.images_dir)
                    if f.lower().endswith(".jpg")
                ]
            )
        else:
            self.file_list = file_list  # Assigns the provided file list.

    def __len__(self) -> int:
        """
        Returns the number of images in the dataset.

        Returns:
            int: The number of images.
        """
        return len(self.file_list)  # Returns the length of the file list.

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Loads a single image and its corresponding OBB annotation (if available).

        This method:
        - Loads the image in RGB format.
        - Parses its label file (.txt) if it exists.
        - Denormalizes the polygon coordinates from [0,1] to absolute pixels.
        - Constructs the target dictionary with 'boxes', 'angles', 'class_idx', and 'valid_mask'.
        - Applies optional transform.

        Args:
            idx (int): Index of the image in the dataset.

        Returns:
            Dict[str, Any]: A dictionary with:
                - "image" (np.ndarray): The image in H×W×3 RGB format.
                - "target" (dict): A dictionary with:
                    - "boxes" (Tensor): (N, 8) absolute polygon vertex coordinates.
                    - "angles" (Tensor): (N,) rotation angles in radians.
                    - "class_idx" (Tensor): (N,) class indices (0 to 4), or 5 if background.
                    - "valid_mask" (Tensor): (N,) boolean mask indicating valid entries.
        """
        base = self.file_list[idx]
        img_path = os.path.join(self.images_dir, base + ".jpg")
        lbl_path = os.path.join(self.labels_dir, base + ".txt")

        # 1) Load RGB image
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]

        boxes: List[List[float]] = []
        angles: List[float] = []
        class_idxs: List[int] = []

        # 2) Parse label file (if it exists)
        if os.path.exists(lbl_path):
            with open(lbl_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 10:
                        continue  # skip malformed lines
                    cls = int(parts[0])
                    coords = list(map(float, parts[1:9]))
                    ang = float(parts[9])

                    # Denormalize coordinates (x1, y1, ..., x4, y4) from [0,1] to absolute pixels
                    pts_px: List[float] = []
                    for i in range(0, 8, 2):
                        x = coords[i] * W
                        y = coords[i + 1] * H
                        pts_px.extend([x, y])

                    class_idxs.append(cls)
                    boxes.append(pts_px)
                    angles.append(ang)

        # 3) Convert lists to tensors (or empty tensors if no GT)
        if len(boxes) > 0:
            boxes_t = torch.tensor(boxes, dtype=torch.float32)  # (N,8)
            angles_t = torch.tensor(angles, dtype=torch.float32)  # (N,)
            cls_t = torch.tensor(class_idxs, dtype=torch.long)  # (N,)
            valid_mask = torch.ones(len(boxes), dtype=torch.bool)  # (N,)
        else:
            # No ground truth available → treat as background
            boxes_t = torch.zeros((0, 8), dtype=torch.float32)
            angles_t = torch.zeros((0,), dtype=torch.float32)
            cls_t = torch.zeros((0,), dtype=torch.long)
            valid_mask = torch.zeros((0,), dtype=torch.bool)

        # 4) Build target dictionary
        target: Dict[str, torch.Tensor] = {
            "boxes": boxes_t,
            "angles": angles_t,
            "class_idx": cls_t,
            "valid_mask": valid_mask,
        }

        # 5) Build sample dictionary
        sample: Dict[str, Any] = {
            "image": img,
            "target": target,
        }

        # 6) Apply transform (if any)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample


def compute_dataset_mean_std(
    dataset: Dataset, max_samples: Optional[int] = None
) -> Tuple[List[float], List[float]]:
    """
    Computes the mean and standard deviation per channel for the given dataset.

    Args:
        dataset (Dataset): A PyTorch dataset returning samples with key "image".
        max_samples (Optional[int]): If specified, limits the number of samples processed.

    Returns:
        tuple: (mean, std) as 3-element lists for RGB channels.
    """
    mean = torch.zeros(
        3
    )  # Initializes a tensor to store the sum of pixel values for each channel.
    std = torch.zeros(
        3
    )  # Initializes a tensor to store the sum of squared pixel values for each channel.
    n_pixels = 0  # Initializes a variable to store the total number of pixels.

    num_samples = (
        len(dataset) if max_samples is None else min(len(dataset), max_samples)
    )  # Determines the number of samples to process.

    for i in range(num_samples):  # Iterates through the specified number of samples.
        sample = dataset[i]  # Retrieves the i-th sample from the dataset.
        image = sample[
            "image"
        ]  # numpy array HxWxC, uint8. Retrieves the image from the sample as a NumPy array.

        # Convert image to float32
        image = (
            torch.from_numpy(image).float() / 255.0
        )  # CxHxW. Converts the image to a float tensor and normalizes it to [0, 1].
        image = image.permute(
            2, 0, 1
        )  # Convert to CxHxW. Permutes the image tensor to have channels first (C, H, W).

        n = (
            image.numel() // 3
        )  # pixels per channel. Calculates the number of pixels per channel.
        mean += image.sum(
            dim=[1, 2]
        )  # Adds the sum of pixel values for each channel to the mean tensor.
        std += (image**2).sum(
            dim=[1, 2]
        )  # Adds the sum of squared pixel values for each channel to the std tensor.
        n_pixels += (
            n  # Adds the number of pixels per channel to the total number of pixels.
        )

    mean /= n_pixels  # Calculates the mean pixel value for each channel.
    std = (
        std / n_pixels - mean**2
    ).sqrt()  # Calculates the standard deviation for each channel.

    return mean.tolist(), std.tolist()


def calculate_average_obb_dimensions(dataset: Dataset, img_size) -> Dict[str, float]:
    """
    Calculates the average size, width, height, and aspect ratio of oriented bounding boxes (OBBs) in a dataset.

    Args:
        dataset (Dataset): A PyTorch dataset where each sample contains OBB annotations in the "target" dictionary.
        img_size (Tuple[int, int]): The target size to which the images are resized.

    Returns:
        Dict[str, float]: A dictionary containing the average OBB size, width, height, and aspect ratio.
            - "avg_size": The average of the average dimensions (width + height) / 2.
            - "avg_width": The average width of the OBBs.
            - "avg_height": The average height of the OBBs.
            - "avg_ratio": The average height-to-width ratio of the OBBs.
    """
    resize_only = Resize(size=img_size)
    sizes = []  # List to store the average dimensions of each OBB.
    widths = []  # List to store the widths of each OBB.
    heights = []  # List to store the heights of each OBB.
    ratios = []  # List to store the aspect ratios (height / width) of each OBB.

    for i in range(len(dataset)):  # Iterates through each sample in the dataset.
        sample = dataset[i]  # Retrieves the i-th sample.
        sample = resize_only(sample)  # Applies the resize transform to the sample.
        for box in sample["target"][
            "boxes"
        ]:  # Iterates through each OBB in the sample.
            pts = box.view(
                4, 2
            )  # Reshapes the OBB tensor to (4, 2) for easier coordinate access.
            w = torch.norm(
                pts[1] - pts[0]
            )  # Calculates the width of the OBB (distance between top-right and top-left points).
            h = torch.norm(
                pts[2] - pts[1]
            )  # Calculates the height of the OBB (distance between bottom-right and top-right points).
            size = (w + h) / 2  # Calculates the average dimension of the OBB.
            sizes.append(
                size.item()
            )  # Appends the average dimension to the sizes list.
            widths.append(w.item())  # Appends the width to the widths list.
            heights.append(h.item())  # Appends the height to the heights list.
            ratios.append(
                (h / w).item()
            )  # Appends the aspect ratio to the ratios list.

    return {
        "avg_size": sum(sizes) / len(sizes),  # Calculates the average OBB size.
        "avg_width": sum(widths) / len(widths),  # Calculates the average OBB width.
        "avg_height": sum(heights) / len(heights),  # Calculates the average OBB height.
        "avg_ratio": sum(ratios)
        / len(ratios),  # Calculates the average OBB aspect ratio.
    }
