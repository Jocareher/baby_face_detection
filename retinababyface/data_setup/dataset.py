import os
import cv2
from typing import List, Optional, Callable, Dict, Any, Tuple

import torch
from torch.utils.data import Dataset


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
        Retrieves an image and its corresponding target (bounding boxes, angles, class indices) given an index.

        Args:
            idx (int): Index of the image to retrieve.

        Returns:
            Dict[str, Any]: A dictionary containing the image and its target.
                - "image": The image as a NumPy array (H, W, C).
                - "target": A dictionary containing the target information:
                    - "boxes": A torch tensor of shape (N, 8) representing the oriented bounding box coordinates in pixel space.
                    - "angles": A torch tensor of shape (N,) representing the rotation angles in radians.
                    - "class_idxs": A torch tensor of shape (N,) or (1,) if background, representing the class indices.
        """
        base_name = self.file_list[
            idx
        ]  # Gets the base name of the image from the file list.
        img_path = os.path.join(
            self.images_dir, base_name + ".jpg"
        )  # Constructs the path to the image.
        label_path = os.path.join(
            self.labels_dir, base_name + ".txt"
        )  # Constructs the path to the label file.

        # Load image in RGB format
        image = cv2.imread(img_path)  # Reads the image using OpenCV.
        if image is None:  # Checks if the image was loaded successfully.
            raise FileNotFoundError(
                f"Could not read image file: {img_path}"
            )  # Raises an error if the image could not be read.
        image = cv2.cvtColor(
            image, cv2.COLOR_BGR2RGB
        )  # Converts the image from BGR to RGB.
        height, width = image.shape[:2]  # Gets the height and width of the image.

        boxes = []  # List of flattened (x1, y1, ..., x4, y4) in pixel coordinates
        angles = []  # Rotation angles in radians
        class_idxs = []  # Class indices

        if os.path.exists(label_path):  # Checks if the label file exists.
            with open(label_path, "r") as f:  # Opens the label file for reading.
                for line in f:  # Iterates over each line in the label file.
                    parts = line.strip().split()  # Splits the line into parts.
                    if (
                        len(parts) != 10
                    ):  # Checks if the line has the correct number of parts.
                        continue  # Skips malformed lines.

                    class_idx = int(parts[0])  # Gets the class index.
                    coords = list(map(float, parts[1:9]))  # Gets the coordinates.
                    angle = float(parts[9])  # Gets the angle.

                    coords_px = []  # List to store coordinates in pixel space.
                    for i in range(0, 8, 2):  # Iterates over the coordinates.
                        x = (
                            coords[i] * width
                        )  # Calculates the x-coordinate in pixel space.
                        y = (
                            coords[i + 1] * height
                        )  # Calculates the y-coordinate in pixel space.
                        coords_px.extend([x, y])  # Adds the coordinates to the list.

                    class_idxs.append(class_idx)  # Adds the class index to the list.
                    boxes.append(coords_px)  # Adds the coordinates to the list.
                    angles.append(angle)  # Adds the angle to the list.
        else:
            # Background image — no boxes or angles, just class index 5
            class_idxs.append(5)  # Adds class index 5 for background images.

        # Build target dictionary
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),  # shape (N, 8) or (0, 8)
            "angles": torch.tensor(angles, dtype=torch.float32),  # shape (N,) or (0,)
            "class_idxs": torch.tensor(
                class_idxs, dtype=torch.long
            ),  # shape (N,) or (1,) if background
        }

        sample = {
            "image": image,
            "target": target,
        }

        if self.transform:
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

        for i in range(
            num_samples
        ):  # Iterates through the specified number of samples.
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
            n_pixels += n  # Adds the number of pixels per channel to the total number of pixels.

        mean /= n_pixels  # Calculates the mean pixel value for each channel.
        std = (
            std / n_pixels - mean**2
        ).sqrt()  # Calculates the standard deviation for each channel.

        return mean.tolist(), std.tolist()

    def calculate_average_obb_dimensions(dataset: Dataset) -> Dict[str, float]:
        """
        Calculates the average size, width, height, and aspect ratio of oriented bounding boxes (OBBs) in a dataset.

        Args:
            dataset (Dataset): A PyTorch dataset where each sample contains OBB annotations in the "target" dictionary.

        Returns:
            Dict[str, float]: A dictionary containing the average OBB size, width, height, and aspect ratio.
                - "avg_size": The average of the average dimensions (width + height) / 2.
                - "avg_width": The average width of the OBBs.
                - "avg_height": The average height of the OBBs.
                - "avg_ratio": The average height-to-width ratio of the OBBs.
        """
        sizes = []  # List to store the average dimensions of each OBB.
        widths = []  # List to store the widths of each OBB.
        heights = []  # List to store the heights of each OBB.
        ratios = []  # List to store the aspect ratios (height / width) of each OBB.

        for i in range(len(dataset)):  # Iterates through each sample in the dataset.
            sample = dataset[i]  # Retrieves the i-th sample.
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
            "avg_height": sum(heights)
            / len(heights),  # Calculates the average OBB height.
            "avg_ratio": sum(ratios)
            / len(ratios),  # Calculates the average OBB aspect ratio.
        }
