import os
from pathlib import Path
from collections import Counter
from typing import List, Optional, Callable, Dict, Any, Tuple

import torch
from torch.utils.data import Dataset
from sklearn.cluster import KMeans
import cv2
import numpy as np
from torch.utils.data import WeightedRandomSampler

from .augmentations import Resize, wrap_to_pi
from loss.utils import xyxyxyxy2xywhr


class BabyFacesDataset(Dataset):
    """
    PyTorch Dataset class for loading baby face images and their associated oriented bounding box (OBB) annotations.

    Each image may have one or more annotations stored in a corresponding .txt label file.
    Label format per line:
        class_idx child_prob x1 y1 x2 y2 x3 y3 x4 y4 angle

    - class_idx: integer from 0 to 4 indicating face orientation
        (0 = left profile, 1 = 3/4 leftside, 2 = frontal, 3 = 3/4 rightside, 4 = right profile)
    - child_prob: integer (0 or 1) indicating if the face is a child
    - x1, y1, ..., x4, y4: normalized (0–1) coordinates of the OBB corners
    - angle: rotation angle in radians (clockwise), usually measured from the top-left corner

    Images without a .txt file yield zero GT boxes; background (no-face)
    handling is done later by anchor matching.

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
        - Constructs the target dictionary with 'boxes', 'angles', 'class_idx', 'child_prob', and 'valid_mask'.
        - Applies optional transform.

        Args:
            idx (int): Index of the image in the dataset.

        Returns:
            Dict[str, Any]: A dictionary with:
                - "image" (np.ndarray): The image in H×W×3 RGB format.
                - "target" (dict): A dictionary with:
                    - "boxes" (Tensor): (N, 8) absolute polygon vertex coordinates.
                    - "angles" (Tensor): (N,) rotation angles in radians.
                    - "class_idx" (Tensor): (N,) class indices (0 to 4).
                    - "child_prob" (Tensor): (N,) child probabilities (0 or 1).
                    - "valid_mask" (Tensor): (N,) boolean mask indicating valid entries.
                    - "has_face": Tensor[1]   # optionally added below
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
        child_probs: List[int] = []

        # 2) Parse label file (if it exists)
        if os.path.exists(lbl_path):
            with open(lbl_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 11:
                        continue  # skip malformed lines
                    cls = int(parts[0])  # class index (0 to 4)
                    child = int(parts[1])  # child probability (0 or 1)
                    coords = list(
                        map(float, parts[2:10])
                    )  # x1, y1, x2, y2, x3, y3, x4, y4
                    ang = float(parts[10])  # angle in radians

                    # Denormalize coordinates (x1, y1, ..., x4, y4) from [0,1] to absolute pixels
                    pts_px: List[float] = []
                    for i in range(0, 8, 2):
                        x = coords[i] * W
                        y = coords[i + 1] * H
                        pts_px.extend([x, y])

                    class_idxs.append(cls)
                    child_probs.append(child)
                    boxes.append(pts_px)
                    angles.append(ang)

        # 3) Convert lists to tensors (or empty tensors if no GT)
        if len(boxes) > 0:
            boxes_t = torch.tensor(boxes, dtype=torch.float32)  # (N,8)
            angles_t = torch.tensor(angles, dtype=torch.float32)  # (N,)
            cls_t = torch.tensor(class_idxs, dtype=torch.long)  # (N,)
            child_t = torch.tensor(child_probs, dtype=torch.float32)
            valid_mask = torch.ones(len(boxes), dtype=torch.bool)  # (N,)
        else:
            # No ground truth available → treat as background
            boxes_t = torch.zeros((0, 8), dtype=torch.float32)
            angles_t = torch.zeros((0,), dtype=torch.float32)
            cls_t = torch.zeros((0,), dtype=torch.long)
            child_t = torch.zeros((0,), dtype=torch.float32)
            valid_mask = torch.zeros((0,), dtype=torch.bool)

        # Normalize angles to [-pi, pi]
        angles_t = wrap_to_pi(angles_t)

        # 4) Build target dictionary
        target: Dict[str, torch.Tensor] = {
            "boxes": boxes_t,
            "angles": angles_t,
            "class_idx": cls_t,
            "child_prob": child_t,
            "valid_mask": valid_mask,
        }

        # Add a boolean indicating if the image has any faces
        target["has_face"] = torch.tensor(len(boxes) > 0, dtype=torch.bool)

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


def compute_wh_kmeans_clusters(
    dataset: Dataset, img_size: Tuple[int, int] = (640, 640), n_clusters: int = 5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes K-Means clustering over normalized width and height values extracted from oriented bounding boxes (OBBs).

    Args:
        dataset (torch.utils.data.Dataset): Dataset containing samples with target OBBs in 'boxes'.
        img_size (Tuple[int, int]): Size of the image (width, height) used to normalize box dimensions.
        n_clusters (int): Number of clusters to form.

    Returns:
        Tuple:
            - clusters (np.ndarray): Array of shape (n_clusters, 2) with normalized (w, h) centroids.
            - scale_factors (np.ndarray): Array of scale factors (sqrt(w * h)) per cluster.
            - ratio_factors (np.ndarray): Array of aspect ratios (h / w) per cluster.
    """
    all_wh: List[torch.Tensor] = []

    for sample in dataset:
        obbs = sample["target"]["boxes"]  # Tensor of shape (N_i, 8)
        _, _, ws, hs, _ = xyxyxyxy2xywhr(
            obbs,
            torch.zeros(obbs.size(0), device=obbs.device),  # dummy angles
            image_size=img_size,
        ).unbind(1)

        norm_ws = ws / img_size[0]
        norm_hs = hs / img_size[1]
        all_wh.append(torch.stack([norm_ws, norm_hs], dim=1).cpu())

    all_wh_np = torch.cat(all_wh, dim=0).numpy()

    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(all_wh_np)
    clusters = kmeans.cluster_centers_

    ratio_factors = clusters[:, 1] / clusters[:, 0]
    scale_factors = np.sqrt(clusters[:, 0] * clusters[:, 1])

    print("Normalized width-height centroids:\n", clusters)
    print("Aspect ratios (h/w):", ratio_factors)
    print("Scales (sqrt(w*h)):", scale_factors)

    return clusters, scale_factors, ratio_factors


def compute_angle_centroids_kmeans(dataset: Dataset, n_angles: int = 7) -> np.ndarray:
    """
    Computes angle centroids using K-Means clustering on unit vectors derived from ground truth angles.

    Args:
        dataset (torch.utils.data.Dataset): Dataset containing samples with target angles and valid masks.
        n_angles (int): Number of angle clusters (i.e., centroids) to compute.

    Returns:
        np.ndarray: Sorted array of centroid angles in radians.
    """
    all_angles: List[np.ndarray] = []

    for sample in dataset:
        angles = sample["target"]["angles"][sample["target"]["valid_mask"]]  # (N_i,)
        all_angles.append(angles.cpu().numpy())

    angles_array = np.concatenate(all_angles)  # Shape: (Total_GT,)

    # Project angles onto unit circle
    angle_points = np.stack(
        [np.cos(angles_array), np.sin(angles_array)], axis=1
    )  # (N, 2)

    # K-Means clustering on unit circle
    kmeans = KMeans(n_clusters=n_angles, random_state=0).fit(angle_points)
    centers = kmeans.cluster_centers_

    # Convert back to angles in radians
    angle_centroids = np.arctan2(centers[:, 1], centers[:, 0])
    angle_centroids = np.sort(angle_centroids)

    print("Recommended angle centroids (radians):", angle_centroids)
    print("Recommended angle centroids (degrees):", np.degrees(angle_centroids))

    return angle_centroids


def compute_class_alpha(dataset: Dataset, num_classes: int) -> torch.Tensor:
    """
    Iterates through all samples in the `dataset`, counts the number of ground truth (GT) instances for each class,
    and returns an alpha tensor of weights for FocalLoss calculated as:
        alpha[c] = median(counts) / counts[c]

    Args:
        dataset (Dataset): Instance of BabyFacesDataset.
        num_classes (int): Number of classes (excluding background).

    Returns:
        torch.Tensor: Alpha tensor of shape (num_classes,), dtype float32.
    """
    # Initialize the counter
    counts = torch.zeros(num_classes, dtype=torch.long)
    for sample in dataset:
        # sample["target"]["class_idx"] contains all the labels for the image
        labels: torch.Tensor = sample["target"]["class_idx"]
        for c in labels.tolist():
            counts[c] += 1

    # Calculate the median of the counts
    counts_f = counts.float()
    med = counts_f.median()

    # Avoid division by zero
    eps = 1e-6
    alpha = med / (counts_f + eps)

    return alpha


def make_balanced_sampler(dataset):
    """
    Creates a WeightedRandomSampler for balancing the dataset based on class frequencies.

    This function reads only the .txt label files associated with the dataset, avoiding the need to load images or apply augmentations.
    It assigns weights inversely proportional to the frequency of the dominant class in each image.

    Args:
        dataset (BabyFacesDataset): An instance of the BabyFacesDataset.

    Returns:
        WeightedRandomSampler: A sampler that balances the dataset based on class frequencies.
    """
    # 1) Build a list of paths to the label files
    #    The dataset exposes root_dir, split, and file_list attributes.
    labels_dir: Path = Path(dataset.root_dir) / dataset.split / "labels"
    label_files: List[Path] = [labels_dir / f"{stem}.txt" for stem in dataset.file_list]

    # 2) Determine the dominant class for each image
    #    If no faces are present, assign -1 as the dominant class.
    dominant: List[int] = []
    for txt in label_files:
        if (
            not txt.exists()
        ):  # If the label file does not exist, treat as background (-1).
            dominant.append(-1)
            continue
        with open(txt, "r") as f:
            line = f.readline().strip()  # Read the first line of the label file.
            if line == "":  # If the file is empty, treat as background (-1).
                dominant.append(-1)
            else:
                cls: int = int(
                    line.split()[0]
                )  # Extract the class index from the first line.
                dominant.append(cls)

    # 3) Compute weights inversely proportional to class frequency
    freq: Counter = Counter(dominant)  # Count occurrences of each class.
    weights: torch.Tensor = torch.tensor(
        [1.0 / freq[c] for c in dominant], dtype=torch.float
    )  # Assign weights based on inverse frequency.

    # 4) Create the WeightedRandomSampler
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def remap_labels_in_dataset(root_dir: str) -> None:
    """
    Remaps the class indices in all .txt label files (OBB annotations) across
    'train', 'val', and 'test' folders of the dataset, following a new class order.

    It creates new subdirectories called 'labels_updated/' inside each partition
    where the updated label files are saved.

    Expected annotation format in each .txt line:
        class_idx x1 y1 x2 y2 x3 y3 x4 y4 angle

    Args:
        root_dir (str): Root path of the dataset. It must contain:
            root_dir/train/labels/
            root_dir/val/labels/
            root_dir/test/labels/
    """
    # Mapping from old class indices to new class indices
    label_index_swap = {
        0: 1,  # 3/4 Leftside → 1
        1: 3,  # 3/4 Rightside → 3
        2: 2,  # Frontal → 2
        3: 0,  # Left Profile → 0
        4: 4,  # Right Profile → 4
    }

    partitions = ["train", "val", "test"]

    for split in partitions:
        labels_dir = Path(root_dir) / split / "labels"
        updated_dir = Path(root_dir) / split / "labels_updated"
        updated_dir.mkdir(parents=True, exist_ok=True)

        for txt_file in labels_dir.glob("*.txt"):
            updated_lines = []

            with open(txt_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 10:
                        print(
                            f"[WARNING] Skipping malformed line in {txt_file.name}: {line}"
                        )
                        continue
                    try:
                        class_old = int(parts[0])
                        class_new = label_index_swap[class_old]
                        updated_line = " ".join([str(class_new)] + parts[1:])
                        updated_lines.append(updated_line)
                    except KeyError:
                        print(
                            f"[ERROR] Unknown class index in {txt_file.name}: {class_old}"
                        )
                        continue

            # Save updated annotation
            updated_path = updated_dir / txt_file.name
            with open(updated_path, "w") as f_out:
                f_out.write("\n".join(updated_lines) + "\n")

    print(
        "All label files successfully remapped and saved to 'labels_updated/' directories."
    )
