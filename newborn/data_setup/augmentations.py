from pathlib import Path
import csv

import cv2
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import Tuple, List, Optional, Dict


class Resize(object):
    """
    Resizes the image to a fixed size and adjusts the bounding boxes accordingly.
    The size is expected to be specified as (width, height).
    """

    def __init__(self, size: Tuple[int, int]):
        """
        Initializes the Resize transform.

        Args:
            size (Tuple[int, int]): The desired (width, height) of the resized image.
        """
        self.size = size  # (width, height)

    def __call__(self, sample: dict) -> dict:
        """
        Applies the resize transform to the given sample.

        Args:
            sample (dict): A dictionary containing the image and target information.

        Returns:
            dict: The transformed sample.
        """
        image, target = (
            sample["image"],
            sample["target"],
        )  # Extracts the image and target from the sample.
        h, w = image.shape[:2]  # Gets the height and width of the image.
        new_w, new_h = self.size  # Gets the new width and height.

        # Resize the image
        image_resized = cv2.resize(
            image, (new_w, new_h)
        )  # Resizes the image using OpenCV.

        # Adjust the boxes: since the coordinates are in pixels,
        # we multiply by the scaling factor in each axis.
        scale_x = new_w / w  # Calculates the scaling factor for the x-axis.
        scale_y = new_h / h  # Calculates the scaling factor for the y-axis.
        boxes = target["boxes"].clone()  # Creates a copy of the bounding boxes tensor.
        # Each row has [x1, y1, x2, y2, x3, y3, x4, y4]
        boxes = boxes.view(-1, 4, 2)  # Reshape to (N, 4, 2) for vectorized scaling.
        boxes[..., 0] *= scale_x  # Scale x-coordinates.
        boxes[..., 1] *= scale_y  # Scale y-coordinates.
        boxes = boxes.view(-1, 8)  # Reshape back to (N, 8)

        target["boxes"] = boxes  # Updates the boxes in the target dictionary.
        sample["image"] = image_resized  # Updates the resized image in the sample.
        sample["target"] = target  # Updates the target in the sample.
        return sample


class RandomHorizontalFlipOBB:
    """
    Applies a horizontal flip to the image and updates:
    - OBB coordinates
    - angles (negated)
    - class indices (0↔4, 1↔3)
    """

    def __init__(self, prob: float = 0.5):
        """
        Initializes the RandomHorizontalFlipOBB transform.

        Args:
            prob (float): The probability of applying the flip. Defaults to 0.5.
        """
        self.prob = prob

    def __call__(self, sample: dict) -> dict:
        """
        Applies the horizontal flip transform to the given sample.

        Args:
            sample (dict): A dictionary containing the image and target information.

        Returns:
            dict: The transformed sample.
        """
        image, target = (
            sample["image"],
            sample["target"],
        )  # Extracts the image and target from the sample.
        if random.random() < self.prob:  # Checks if the flip should be applied.
            h, w = image.shape[:2]  # Gets the height and width of the image.
            # Flip image
            image = np.fliplr(image).copy()  # Flips the image horizontally.

            boxes = target[
                "boxes"
            ].clone()  # Creates a copy of the bounding boxes tensor.
            angles = target["angles"].clone()  # Creates a copy of the angles tensor.
            class_idxs = target[
                "class_idx"
            ].clone()  # Creates a copy of the class indices tensor.
            child_prob = target[
                "child_prob"
            ].clone()  # Creates a copy of the child probabilities tensor.

            # Flip X coordinates
            boxes = boxes.view(-1, 4, 2)  # (N, 4, 2)
            boxes[..., 0] = w - boxes[..., 0]  # Flip x-coordinates

            # Reorder points to maintain orientation: [1, 0, 3, 2]
            reorder_idx = torch.tensor([1, 0, 3, 2], device=boxes.device)
            boxes = boxes[:, reorder_idx, :]
            boxes = boxes.view(-1, 8)  # Back to (N, 8)

            #  Negate & wrap into [-π,π]
            angles = wrap_to_pi(-angles)

            # Flip class indices: vectorized swap using masks
            class_idxs_flipped = class_idxs.clone()
            swap_map = {0: 4, 1: 3, 3: 1, 4: 0}
            for a, b in swap_map.items():
                class_idxs_flipped[class_idxs == a] = b

            target["boxes"] = boxes  # Updates the boxes in the target dictionary.
            target["angles"] = angles  # Updates the angles in the target dictionary.
            target[
                "class_idx"
            ] = class_idxs_flipped  # Updates the class indices in the target dictionary.
            target[
                "child_prob"
            ] = child_prob  # Updates the child probabilities in the target dictionary.

        sample["image"] = image  # Updates the image in the sample.
        sample["target"] = target  # Updates the target in the sample.
        return sample


class RandomRotateOBB:
    """
    Randomly rotates the image and OBBs by an angle in degrees between [-max_angle, max_angle],
    expanding the canvas to avoid cropping, and normalizing the resulting angles.
    """

    def __init__(self, max_angle: int = 30, prob: float = 0.5):
        """
        Initializes the RandomRotateOBB transform.

        Args:
            max_angle (int): The maximum rotation angle in degrees. Defaults to 180.
            prob (float): The probability of applying the rotation. Defaults to 0.5.
        """
        self.max_angle = max_angle
        self.prob = prob

    def __call__(self, sample: dict) -> dict:
        """
        Applies the random rotation transform to the given sample.

        Args:
            sample (dict): A dictionary containing the image and target information.

        Returns:
            dict: The transformed sample.
        """
        if random.random() > self.prob:  # Checks if the rotation should be applied.
            return sample

        image, target = (
            sample["image"],
            sample["target"],
        )  # Extracts the image and target from the sample.
        h, w = image.shape[:2]  # Gets the height and width of the image.
        angle_deg = random.uniform(
            -self.max_angle, self.max_angle
        )  # Generates a random rotation angle in degrees (remove (-) sign).
        angle_rad = np.radians(angle_deg)  # Converts the angle to radians.

        # Compute new canvas size
        abs_cos = abs(
            math.cos(angle_rad)
        )  # Calculates the absolute cosine of the angle.
        abs_sin = abs(math.sin(angle_rad))  # Calculates the absolute sine of the angle.
        new_w = int(
            h * abs_sin + w * abs_cos
        )  # Calculates the new width of the canvas.
        new_h = int(
            h * abs_cos + w * abs_sin
        )  # Calculates the new height of the canvas.

        # Compute rotation matrix and adjust for canvas shift
        center = (w / 2, h / 2)  # Calculates the center of the image.
        rot_mat = cv2.getRotationMatrix2D(
            center, angle_deg, 1.0
        )  # Gets the rotation matrix.
        rot_mat[0, 2] += (
            new_w - w
        ) / 2  # Adjusts the rotation matrix for the canvas shift.
        rot_mat[1, 2] += (
            new_h - h
        ) / 2  # Adjusts the rotation matrix for the canvas shift.

        # Rotate image with expanded canvas
        rotated_image = cv2.warpAffine(
            image, rot_mat, (new_w, new_h), flags=cv2.INTER_LINEAR
        )  # Rotates the image.

        boxes = target["boxes"].clone()  # Creates a copy of the bounding boxes tensor.
        angles = target["angles"].clone()  # Creates a copy of the angles tensor.
        class_idxs = target[
            "class_idx"
        ].clone()  # Creates a copy of the class indices tensor.
        child_prob = target[
            "child_prob"
        ].clone()  # Creates a copy of the child probabilities tensor.

        # Vectorized rotation of all boxes
        N = boxes.shape[0]
        boxes_np = boxes.view(N, 4, 2).cpu().numpy()  # (N, 4, 2)
        ones = np.ones((N, 4, 1), dtype=np.float32)
        boxes_hom = np.concatenate([boxes_np, ones], axis=2)  # (N, 4, 3)
        rot_mat_np = rot_mat.astype(np.float32)

        rotated_boxes = np.matmul(boxes_hom, rot_mat_np.T)  # (N, 4, 2)
        boxes = torch.tensor(
            rotated_boxes.reshape(N, 8),
            dtype=torch.float32,
            device=target["boxes"].device,
        )

        # Subtract rotation & wrap into [-π,π]
        angles = wrap_to_pi(angles - angle_rad)

        target["boxes"] = boxes  # Updates the boxes in the target dictionary.
        target["angles"] = angles  # Updates the angles in the target dictionary.
        target[
            "class_idx"
        ] = class_idxs  # Updates the class indices in the target dictionary.
        target[
            "child_prob"
        ] = child_prob  # Updates the child probabilities in the target dictionary.

        sample["image"] = rotated_image  # Updates the rotated image in the sample.
        sample["target"] = target  # Updates the target in the sample.

        # Add a valid mask
        num = target["boxes"].shape[0]
        # Create a valid mask for the boxes
        target["valid_mask"] = torch.ones(num, dtype=torch.bool, device=boxes.device)
        # If there are no boxes, set the valid mask to False
        sample["target"] = target
        return sample


class RandomScaleTranslateOBB:
    """
    Randomly scales and translates the image and its OBBs.
    Canvas is expanded to avoid cropping. OBBs completely outside the frame are removed.
    """

    def __init__(
        self,
        scale_range: Tuple[float, float] = (0.85, 1.15),
        translate_range: Tuple[float, float] = (-0.1, 0.1),
        prob: float = 0.5,
    ):
        """
        Initializes the RandomScaleTranslateOBB transform.

        Args:
            scale_range (Tuple[float, float]): The range of scaling factors. Defaults to (0.85, 1.15).
            translate_range (Tuple[float, float]): The range of translation factors. Defaults to (-0.1, 0.1).
            prob (float): The probability of applying the transform. Defaults to 0.5.
        """
        self.scale_range = scale_range
        self.translate_range = translate_range
        self.prob = prob

    def __call__(self, sample: dict) -> dict:
        """
        Applies the random scale and translate transform to the given sample.
        Args:
            sample (dict): A dictionary containing the image and target information.
        Returns:
            dict: The transformed sample.
        """
        # Checks if the transform should be applied.
        if random.random() > self.prob:
            return sample

        # Extracts the image and target from the sample.
        image, target = sample["image"], sample["target"]
        # Gets the height and width of the image.
        h, w = image.shape[:2]
        # Generates a random scaling factor and translation factors.
        scale = random.uniform(*self.scale_range)
        # Calculates the translation factors based on the scaling factor.
        tx = random.uniform(*self.translate_range) * w
        ty = random.uniform(*self.translate_range) * h

        # Calculates the new width and height of the image.
        new_w = int(w * scale + abs(tx))
        new_h = int(h * scale + abs(ty))
        # Calculates the center of the image.
        shift_x = max(tx, -tx)
        shift_y = max(ty, -ty)
        # Creates the transformation matrix for scaling and translation.
        M = np.array([[scale, 0, shift_x], [0, scale, shift_y]], dtype=np.float32)

        # Applies the transformation matrix to the image.
        transformed_image = cv2.warpAffine(
            image, M, (new_w, new_h), flags=cv2.INTER_LINEAR
        )
        # Adjusts the boxes: since the coordinates are in pixels,
        # we multiply by the scaling factor in each axis.
        boxes = target["boxes"]
        angles = target["angles"]
        class_idxs = target["class_idx"]
        child_prob = target["child_prob"]

        # If there are no boxes, return the transformed image and target.
        if boxes.shape[0] == 0:
            sample["image"], sample["target"] = transformed_image, target
            return sample

        # Vectorized rotation of all boxes
        N = boxes.shape[0]
        # Creates a copy of the bounding boxes tensor.
        pts = boxes.view(N, 4, 2).cpu().numpy()  # (N,4,2)
        ones = np.ones((N, 4, 1), dtype=np.float32)
        hom = np.concatenate([pts, ones], axis=2)  # (N,4,3)
        pts_t = hom @ M.T  # (N,4,2)

        # Clip the points to the new image size
        pts_t[..., 0] = np.clip(pts_t[..., 0], 0, new_w - 1)
        pts_t[..., 1] = np.clip(pts_t[..., 1], 0, new_h - 1)

        # Check if any points are inside the new image size
        inside_x = (0 <= pts_t[..., 0]) & (pts_t[..., 0] < new_w)
        inside_y = (0 <= pts_t[..., 1]) & (pts_t[..., 1] < new_h)
        valid = (inside_x | inside_y).any(axis=1)

        # If no points are valid, return the transformed image and target
        if not valid.any():
            return sample

        # Keep only the valid points
        keep_pts = pts_t[valid]
        # Keep only the valid angles
        keep_angles = wrap_to_pi(angles[valid])
        # Keep only the valid class indices
        keep_classes = class_idxs[valid]
        # Keep only the valid child probabilities
        keep_child_prob = child_prob[valid]

        # Reshape the points to (N,8)
        target["boxes"] = torch.tensor(
            keep_pts.reshape(-1, 8), dtype=torch.float32, device=boxes.device
        )
        target["angles"] = keep_angles
        target["class_idx"] = keep_classes
        target["child_prob"] = keep_child_prob
        target["valid_mask"] = torch.ones(
            len(keep_classes), dtype=torch.bool, device=boxes.device
        )

        # Update the target with the new boxes, angles, and class indices
        sample["image"], sample["target"] = transformed_image, target
        return sample


class ColorJitterOBB:
    """
    Randomly changes the brightness, contrast, and saturation of the image.
    This does not affect the OBBs or angles.
    """

    def __init__(
        self,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.05,
        gamma: float = 0.1,
        prob: float = 0.8,
    ):
        """
        Initializes the ColorJitterOBB transform.

        Args:
            brightness (float): The brightness adjustment factor. Defaults to 0.2.
            contrast (float): The contrast adjustment factor. Defaults to 0.2.
            saturation (float): The saturation adjustment factor. Defaults to 0.2.
            hue (float): The hue adjustment factor. Defaults to 0.05.
            gamma (float): The gamma adjustment factor. Defaults to 0.1.
            prob (float): The probability of applying the transform. Defaults to 0.8.
        """
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.gamma = gamma
        self.prob = prob

    def __call__(self, sample: dict) -> dict:
        """
        Applies the color jitter transform to the given sample.

        Args:
            sample (dict): A dictionary containing the image and target information.

        Returns:
            dict: The transformed sample.
        """
        if random.random() > self.prob:  # Checks if the transform should be applied.
            return sample

        image = sample["image"].astype(np.float32)  # Converts the image to float32.

        # Brightness
        if self.brightness > 0:  # Checks if brightness adjustment should be applied.
            factor = 1.0 + random.uniform(
                -self.brightness, self.brightness
            )  # Generates a random brightness factor.
            image *= factor  # Adjusts the brightness of the image.

        # Contrast
        if self.contrast > 0:  # Checks if contrast adjustment should be applied.
            mean = np.mean(
                image, axis=(0, 1), keepdims=True
            )  # Calculates the mean of the image.
            factor = 1.0 + random.uniform(
                -self.contrast, self.contrast
            )  # Generates a random contrast factor.
            image = (image - mean) * factor + mean  # Adjusts the contrast of the image.

        # Saturation (only affects RGB, so convert to HSV)
        if self.saturation > 0:  # Checks if saturation adjustment should be applied.
            image_uint8 = np.clip(image, 0, 255).astype(
                np.uint8
            )  # Clip before HSV conversion.
            hsv = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2HSV).astype(
                np.float32
            )  # Converts the image to HSV.
            factor = 1.0 + random.uniform(
                -self.saturation, self.saturation
            )  # Generates a random saturation factor.
            hsv[..., 1] *= factor  # Adjusts the saturation of the image.
            hsv[..., 1] = np.clip(
                hsv[..., 1], 0, 255
            )  # Clips the saturation values to [0, 255].
            image = cv2.cvtColor(
                hsv.astype(np.uint8), cv2.COLOR_HSV2RGB
            )  # Converts the image back to RGB.
        else:
            image = np.clip(image, 0, 255).astype(
                np.uint8
            )  # Ensures the image is valid if saturation not applied.

        # Hue (only affects HSV, so convert to HSV)
        if self.hue > 0:
            # Convert to HSV
            delta = random.uniform(-self.hue * 180, self.hue * 180)
            hsv[..., 0] = (hsv[..., 0] + delta) % 180
        image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        # Gamma
        if self.gamma > 0:
            # Apply gamma correction
            # Generate a random gamma factor
            # between (1-gamma, 1+gamma)
            # and apply it to the image
            # to simulate low-quality or motion blur
            gamma_factor = 1.0 + random.uniform(-self.gamma, self.gamma)
            inv = 1.0 / gamma_factor
            table = np.array([(i / 255.0) ** inv * 255 for i in np.arange(256)]).astype(
                "uint8"
            )
            image = cv2.LUT(image.astype(np.uint8), table)

        sample["image"] = image  # Updates the image in the sample.
        return sample


class RandomNoiseOBB:
    """
    Adds random Gaussian noise to the image.
    """

    def __init__(self, std: float = 10, prob: float = 0.5):
        """
        Initializes the RandomNoiseOBB transform.

        Args:
            std (float): The standard deviation of the Gaussian noise. Defaults to 10.
            prob (float): The probability of applying the transform. Defaults to 0.5.
        """
        self.std = std
        self.prob = prob

    def __call__(self, sample: dict) -> dict:
        """
        Applies the random noise transform to the given sample.

        Args:
            sample (dict): A dictionary containing the image and target information.

        Returns:
            dict: The transformed sample.
        """
        if random.random() > self.prob:  # Checks if the transform should be applied.
            return sample

        image = sample["image"].astype(np.float32)  # Converts the image to float32.
        noise = np.random.normal(loc=0.0, scale=self.std, size=image.shape).astype(
            np.float32
        )  # Generates random Gaussian noise.
        image = np.clip(image + noise, 0, 255).astype(
            np.uint8
        )  # Adds the noise to the image and clips the values to [0, 255].

        sample["image"] = image  # Updates the image in the sample.
        return sample


class RandomBlurOBB:
    """
    Applies Gaussian blur to simulate low-quality or motion blur.
    """

    def __init__(self, ksize: Tuple[int, int] = (5, 5), prob: float = 0.5):
        """
        Initializes the RandomBlurOBB transform.

        Args:
            ksize (Tuple[int, int]): The kernel size for the Gaussian blur. Defaults to (5, 5).
            prob (float): The probability of applying the transform. Defaults to 0.5.
        """
        self.ksize = ksize
        self.prob = prob

    def __call__(self, sample: dict) -> dict:
        """
        Applies the random blur transform to the given sample.

        Args:
            sample (dict): A dictionary containing the image and target information.

        Returns:
            dict: The transformed sample.
        """
        if random.random() > self.prob:  # Checks if the transform should be applied.
            return sample

        image = sample["image"]  # Gets the image from the sample.
        image = cv2.GaussianBlur(
            image, self.ksize, sigmaX=0
        )  # Applies Gaussian blur to the image.

        sample["image"] = image  # Updates the blurred image in the sample.
        return sample


class RandomOcclusionOBB:
    """
    Randomly occludes a rectangular area inside an OBB (or anywhere in the image).
    If `target_inside_obb=True`, the occlusion is constrained to the selected OBB area.
    The patch size is calculated relative to the size of the OBB instead of the whole image.
    """

    def __init__(
        self,
        max_size_ratio: float = 0.5,
        prob: float = 0.5,
        target_inside_obb: bool = True,
    ):
        """
        Initializes the RandomOcclusionOBB transform.

        Args:
            max_size_ratio (float): The maximum size ratio of the occlusion patch. Defaults to 0.5.
            prob (float): The probability of applying the transform. Defaults to 0.5.
            target_inside_obb (bool): Whether to constrain the occlusion to the OBB area. Defaults to True.
        """
        self.max_size_ratio = max_size_ratio
        self.prob = prob
        self.target_inside_obb = target_inside_obb

    def __call__(self, sample: dict) -> dict:
        """
        Applies the random occlusion transform to the given sample.

        Args:
            sample (dict): A dictionary containing the image and target information.

        Returns:
            dict: The transformed sample.
        """
        if random.random() > self.prob:  # Checks if the transform should be applied.
            return sample

        image = sample["image"]  # Gets the image from the sample.
        h, w = image.shape[:2]  # Gets the height and width of the image.

        if (
            self.target_inside_obb
            and "boxes" in sample["target"]
            and len(sample["target"]["boxes"]) > 0
        ):  # Checks if the occlusion should be inside an OBB.
            # Select a random OBB
            obb = (
                sample["target"]["boxes"][
                    random.randint(0, len(sample["target"]["boxes"]) - 1)
                ]
                .view(4, 2)
                .numpy()
            )  # Selects a random OBB.
            x_min, y_min = obb.min(
                axis=0
            )  # Gets the minimum x and y coordinates of the OBB.
            x_max, y_max = obb.max(
                axis=0
            )  # Gets the maximum x and y coordinates of the OBB.
            obb_w = max(x_max - x_min, 1)  # Calculates the width of the OBB.
            obb_h = max(y_max - y_min, 1)  # Calculates the height of the OBB.

            occ_w = int(
                random.uniform(0.1, self.max_size_ratio) * obb_w
            )  # Calculates the width of the occlusion patch.
            occ_h = int(
                random.uniform(0.1, self.max_size_ratio) * obb_h
            )  # Calculates the height of the occlusion patch.

            # Clamp occlusion position within the OBB bounding rectangle
            x0_min = int(max(x_min, 0))
            y0_min = int(max(y_min, 0))
            x0_max = max(int(x_max - occ_w), x0_min + 1)
            y0_max = max(int(y_max - occ_h), y0_min + 1)

            x0 = random.randint(x0_min, x0_max)
            y0 = random.randint(y0_min, y0_max)
        else:
            # Occlusion anywhere in the image
            occ_w = int(
                random.uniform(0.1, self.max_size_ratio) * w
            )  # Calculates the width of the occlusion patch.
            occ_h = int(
                random.uniform(0.1, self.max_size_ratio) * h
            )  # Calculates the height of the occlusion patch.
            x0 = random.randint(
                0, max(w - occ_w, 1)
            )  # Generates a random x coordinate for the occlusion.
            y0 = random.randint(
                0, max(h - occ_h, 1)
            )  # Generates a random y coordinate for the occlusion.

        # Apply occlusion
        image[
            y0 : y0 + occ_h, x0 : x0 + occ_w
        ] = 0  # Applies the occlusion to the image.
        sample["image"] = image  # Updates the image in the sample.
        return sample


class RandomGrayOBB:
    """
    Converts the image to grayscale with a certain probability.
    """

    def __init__(self, prob=0.1):
        """
        Initializes the RandomGrayOBB transform.
        Args:
            prob (float): The probability of applying the transform. Defaults to 0.1.
        """
        # Checks if the transform should be applied.
        self.prob = prob

    def __call__(self, sample):
        """
        Applies the random gray transform to the given sample.
        Args:
            sample (dict): A dictionary containing the image and target information.
        Returns:
            dict: The transformed sample.
        1) Converts the image to grayscale with a certain probability.
        2) Stacks the grayscale image to create a 3-channel image.
        3) Updates the image in the sample.
        """
        # Checks if the transform should be applied.
        # If the probability is greater than a random number, apply the transform.
        # Otherwise, return the sample without any changes.
        if random.random() > self.prob:
            return sample
        img = sample["image"]
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        sample["image"] = np.stack([gray] * 3, axis=-1)
        return sample


class ToTensorNormalize(object):
    """
    Converts the image (HxWxC numpy array) to a PyTorch tensor (CxHxW)
    and normalizes it using provided mean and std values.
    """

    def __init__(
        self,
        mean: Tuple[float, float, float] = (
            0.6427208185195923,
            0.5918306708335876,
            0.5525837540626526,
        ),
        std: Tuple[float, float, float] = (
            0.2812318801879883,
            0.28248199820518494,
            0.3035854697227478,
        ),
    ):
        """
        Initializes the ToTensorNormalize transform.

        Args:
            mean (Tuple[float, float, float]): The mean values for normalization. Defaults to (0.6427208185195923, 0.5918306708335876, 0.5525837540626526).
            std (Tuple[float, float, float]): The standard deviation values for normalization. Defaults to (0.2812318801879883, 0.28248199820518494, 0.3035854697227478).
        """
        self.mean = torch.tensor(mean).view(
            3, 1, 1
        )  # Creates a tensor for the mean values.
        self.std = torch.tensor(std).view(
            3, 1, 1
        )  # Creates a tensor for the standard deviation values.

    def __call__(self, sample: dict) -> dict:
        """
        Applies the to tensor and normalize transform to the given sample.

        Args:
            sample (dict): A dictionary containing the image and target information.

        Returns:
            dict: The transformed sample.
        """
        image, target = (
            sample["image"],
            sample["target"],
        )  # Gets the image and target from the sample.
        image = (
            torch.from_numpy(image).permute(2, 0, 1).float().div(255.0)
        )  # Converts the image to a PyTorch tensor and normalizes it.
        image = (
            image - self.mean
        ) / self.std  # Normalizes the image using the mean and standard deviation.
        sample["image"] = image  # Updates the normalized image in the sample.
        return sample


def wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    """
    Wraps an angle in radians to the range [-π, π].
    This is useful for ensuring that angles are within a standard range.

    Args:
        angle (torch.Tensor): The angle in radians to be wrapped.

    Returns:
        torch.Tensor: The wrapped angle in radians.

    """
    return (angle + math.pi) % (2 * math.pi) - math.pi

class RandomRotateOBBEqualizeBins:
    """
    Rota la imagen para que una GT caiga en un bin objetivo.
    strategy: "uniform" | "inverse_freq"
    - "inverse_freq" espera bin_weights (conteos por bin) y pondera por 1/weight.
    ref_policy: "random" | "largest" | "first"
    """

    def __init__(
        self,
        bin_deg: int = 10,
        max_angle: float = 30,
        prob: float = 0.8,
        strategy: str = "uniform",
        bin_weights: Optional[List[int]] = None,
        ref_policy: str = "random",
        max_tries: int = 20,
    ):
        assert 0 < bin_deg <= 180
        assert strategy in ("uniform", "inverse_freq")
        assert ref_policy in ("random", "largest", "first")
        self.bin_deg = bin_deg
        self.max_angle = float(max_angle)
        self.prob = prob
        self.strategy = strategy
        self.ref_policy = ref_policy
        self.max_tries = max_tries

        edges = np.arange(0, 180 + bin_deg, bin_deg, dtype=np.float32)
        self.bin_edges = edges
        self.bin_centers_deg = (edges[:-1] + edges[1:]) * 0.5
        self.K = len(self.bin_centers_deg)

        if strategy == "inverse_freq" and bin_weights is not None:
            w = np.asarray(bin_weights, dtype=np.float64)
            w = np.maximum(w, 1e-8)
            p = (1.0 / w); p = p / p.sum()
            self.prob_bins = p
        else:
            self.prob_bins = np.ones(self.K, dtype=np.float64) / self.K

    @staticmethod
    def _wrap_to_pi(x):
        return (x + np.pi) % (2 * np.pi) - np.pi

    def _choose_ref_index(self, boxes: torch.Tensor, policy: str) -> int:
        N = boxes.shape[0]
        if N == 0: return -1
        if policy == "first": return 0
        if policy == "random": return int(np.random.randint(0, N))
        # largest (área del AABB del OBB)
        pts = boxes.view(N, 4, 2).cpu().numpy()
        w = pts[...,0].max(1) - pts[...,0].min(1)
        h = pts[...,1].max(1) - pts[...,1].min(1)
        return int(np.argmax(w * h))

    def _delta_to_bin_center(self, theta_rad: float, center_deg: float) -> float:
        trg = np.deg2rad(center_deg)
        cands = np.array([self._wrap_to_pi(theta_rad - trg),
                          self._wrap_to_pi(theta_rad + trg)], dtype=np.float32)
        return float(cands[np.argmin(np.abs(cands))])

    def __call__(self, sample: dict) -> dict:
        if np.random.rand() > self.prob:
            return sample

        image, target = sample["image"], sample["target"]
        boxes = target.get("boxes"); angles = target.get("angles")
        if boxes is None or angles is None or boxes.numel() == 0:
            phi = np.deg2rad(np.random.uniform(-self.max_angle, self.max_angle))
            return self._apply_rotation(sample, phi)

        ref_idx = self._choose_ref_index(boxes, self.ref_policy)
        if ref_idx < 0:
            phi = np.deg2rad(np.random.uniform(-self.max_angle, self.max_angle))
            return self._apply_rotation(sample, phi)

        theta_ref = float(angles[ref_idx].item())
        lim = np.deg2rad(self.max_angle) + 1e-6
        for _ in range(self.max_tries):
            k = np.random.choice(self.K, p=self.prob_bins)
            delta = self._delta_to_bin_center(theta_ref, float(self.bin_centers_deg[k]))
            if abs(delta) <= lim:
                return self._apply_rotation(sample, delta)

        # fallback
        phi = np.deg2rad(np.random.uniform(-self.max_angle, self.max_angle))
        return self._apply_rotation(sample, phi)

    def _apply_rotation(self, sample: dict, angle_rad: float) -> dict:
        image, target = sample["image"], sample["target"]
        h, w = image.shape[:2]
        ang_deg = float(np.rad2deg(angle_rad))

        c, s = abs(np.cos(angle_rad)), abs(np.sin(angle_rad))
        new_w, new_h = int(h * s + w * c), int(h * c + w * s)

        rot = cv2.getRotationMatrix2D((w/2.0, h/2.0), ang_deg, 1.0).astype(np.float32)
        rot[0,2] += (new_w - w) / 2.0
        rot[1,2] += (new_h - h) / 2.0

        img_rot = cv2.warpAffine(image, rot, (new_w, new_h), flags=cv2.INTER_LINEAR)

        boxes = target["boxes"].clone(); N = boxes.shape[0]
        if N > 0:
            pts = boxes.view(N,4,2).cpu().numpy().astype(np.float32)
            hom = np.concatenate([pts, np.ones((N,4,1), np.float32)], axis=2)
            pts_rot = hom @ rot.T
            target["boxes"] = torch.tensor(pts_rot.reshape(N,8),
                                           dtype=torch.float32, device=boxes.device)
            ang = target["angles"].clone()
            ang = wrap_to_pi(ang - float(angle_rad))
            target["angles"] = ang

        sample["image"] = img_rot
        target["valid_mask"] = torch.ones(N, dtype=torch.bool, device=target["boxes"].device) if N>0 else torch.zeros(0, dtype=torch.bool)
        sample["target"] = target
        return sample


def angles_rad_to_deg_0_180(t: torch.Tensor) -> np.ndarray:
    t = (t + math.pi) % (2*math.pi) - math.pi
    t = t.abs()
    deg = t * (180.0 / math.pi)
    return torch.clamp(deg, max=180.0 - 1e-6).cpu().numpy()

def collect_deg_by_class_from_dataset(ds, labels_map: Dict[int,str]) -> Dict[str, object]:
    """Itera el dataset (tal cual está definido) y devuelve:
       {'all': [deg...], 'per_cls': {c:[deg...]}, 'counts': {c:int}}"""
    per_cls = {c: [] for c in labels_map.keys()}
    all_deg = []
    for i in range(len(ds)):
        sample = ds[i]
        ang = sample["target"]["angles"]            # (N,)
        cls = sample["target"]["class_idx"]         # (N,)
        mask = sample["target"]["valid_mask"] if "valid_mask" in sample["target"] else torch.ones_like(cls,dtype=torch.bool)
        if ang.numel()==0: continue
        ang = ang[mask]; cls = cls[mask]
        deg = angles_rad_to_deg_0_180(ang)
        all_deg.extend(deg.tolist())
        for d, c in zip(deg, cls.tolist()):
            if c in per_cls: per_cls[c].append(float(d))
    counts = {c: len(per_cls[c]) for c in labels_map.keys()}
    return {"all": all_deg, "per_cls": per_cls, "counts": counts}

def plot_histograms_split(data: Dict[str,object], labels_map: Dict[int,str], bin_deg:int, out_dir: Path, tag:str):
    out_dir.mkdir(parents=True, exist_ok=True)
    # ALL
    bins = np.arange(0, 180 + bin_deg, bin_deg)
    fig, ax = plt.subplots(figsize=(8,4.5))
    ax.hist(data["all"], bins=bins, edgecolor="black")
    ax.set_title(f"{tag}: GT angle histogram (ALL) — bin={bin_deg}°")
    ax.set_xlabel("GT angle [deg]"); ax.set_ylabel("Count"); ax.grid(axis="y", linestyle=":", alpha=0.6)
    for s in ("top","right"): ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(out_dir / f"{tag}_ALL_bin{bin_deg}.png", dpi=200); plt.close(fig)

    # per class
    classes = list(labels_map.keys()); n_cls = len(classes)
    n_cols = min(3, n_cls); n_rows = int(math.ceil(n_cls / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 3.8*n_rows)); axes = np.atleast_2d(axes)
    for i,c in enumerate(classes):
        r, col = divmod(i, n_cols); ax = axes[r, col]
        ax.hist(data["per_cls"][c], bins=bins, edgecolor="black")
        ax.set_title(f"{labels_map[c]} (n={len(data['per_cls'][c])}) — bin={bin_deg}°")
        ax.set_xlabel("GT angle [deg]"); ax.set_ylabel("Count"); ax.grid(axis="y", linestyle=":", alpha=0.6)
        for s in ("top","right"): ax.spines[s].set_visible(False)
    for k in range(n_cls, n_rows*n_cols):
        r, col = divmod(k, n_cols); axes[r,col].axis("off")
    fig.suptitle(f"{tag}: GT angle histogram per class — bin={bin_deg}°")
    fig.tight_layout(rect=[0,0,1,0.97])
    fig.savefig(out_dir / f"{tag}_perclass_bin{bin_deg}.png", dpi=200); plt.close(fig)

def build_bin_weights_from_degrees(all_deg: List[float], bin_deg:int) -> List[int]:
    bins = np.arange(0, 180 + bin_deg, bin_deg)
    counts, _ = np.histogram(all_deg, bins=bins)
    return counts.tolist()  # len = 180/bin_deg

def save_counts_csv(path: Path, stats: Dict[str,object], bin_deg:int, labels_map: Dict[int,str]):
    bins = np.arange(0, 180 + bin_deg, bin_deg)
    rows = []
    counts_all, edges = np.histogram(stats["all"], bins=bins)
    for i,c in enumerate(counts_all):
        rows.append({"scope":"ALL","class_idx":"ALL","class_name":"ALL",
                        "bin_left":int(edges[i]),"bin_right":int(edges[i+1]),"count":int(c)})
    for c, name in labels_map.items():
        counts_c,_ = np.histogram(stats["per_cls"][c], bins=bins)
        for i,cnt in enumerate(counts_c):
            rows.append({"scope":"PERCLASS","class_idx":c,"class_name":name,
                            "bin_left":int(edges[i]),"bin_right":int(edges[i+1]),"count":int(cnt)})
    with open(path,"w",newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)