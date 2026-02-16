import cv2
import math
from pathlib import Path
import csv
import random
import numpy as np
import torch
from typing import Tuple, List, Optional, Sequence, Any, Dict


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


def wrap_to_pi(angle):
    """
    Wraps an angle in radians to the range [-π, π].
    This is useful for ensuring that angles are within a standard range.

    Args:
        angle (torch.Tensor): The angle in radians to be wrapped.

    Returns:
        torch.Tensor: The wrapped angle in radians.

    """
    return (angle + math.pi) % (2 * math.pi) - math.pi


def angles_rad_to_deg_0_180(angles_rad: torch.Tensor) -> np.ndarray:
    """
    Convert angles in radians to degrees in [0, 180).

    The conversion:
      1) Wrap to [-pi, pi]
      2) Take absolute value to map to [0, pi]
      3) Convert to degrees
      4) Clamp to < 180 to avoid falling on the right edge of histogram bins

    Args:
        angles_rad: Tensor of angles in radians.

    Returns:
        NumPy array of angles in degrees in [0, 180).
    """
    wrapped = wrap_to_pi(angles_rad).abs()
    deg = wrapped * (180.0 / math.pi)
    return torch.clamp(deg, max=180.0 - 1e-6).cpu().numpy()


class RandomRotateOBBEqualizeBins:
    """
    Rotate an image so that a chosen ground-truth (GT) OBB angle is moved toward a target bin.

    The target bin is sampled either:
      - uniformly across bins ("uniform"), or
      - inversely proportional to observed bin counts ("inverse_freq").

    Notes:
      - Angles are assumed to be in radians.
      - The transform updates both OBB corner coordinates and `target["angles"]`.

    Args:
        bin_deg: Bin width in degrees over [0, 180].
        max_angle: Maximum absolute rotation angle allowed (in degrees).
        prob: Probability of applying the transform.
        strategy: "uniform" or "inverse_freq".
        bin_weights: Bin counts used when strategy="inverse_freq" (length = 180/bin_deg).
        ref_policy: How to choose the reference GT among N boxes: "random", "largest", or "first".
        max_tries: Maximum attempts to find a bin rotation within the allowed max rotation.
    """

    def __init__(
        self,
        bin_deg: int = 10,
        max_angle: float = 30.0,
        prob: float = 0.8,
        strategy: str = "uniform",
        bin_weights: Optional[Sequence[int]] = None,
        ref_policy: str = "random",
        max_tries: int = 20,
    ) -> None:
        if not (0 < bin_deg <= 180):
            raise ValueError("bin_deg must be in (0, 180].")
        if strategy not in ("uniform", "inverse_freq"):
            raise ValueError("strategy must be 'uniform' or 'inverse_freq'.")
        if ref_policy not in ("random", "largest", "first"):
            raise ValueError("ref_policy must be 'random', 'largest', or 'first'.")
        if max_tries <= 0:
            raise ValueError("max_tries must be > 0.")

        self.bin_deg = int(bin_deg)
        self.max_angle = float(max_angle)
        self.prob = float(prob)
        self.strategy = strategy
        self.ref_policy = ref_policy
        self.max_tries = int(max_tries)

        edges = np.arange(0, 180 + self.bin_deg, self.bin_deg, dtype=np.float32)
        self.bin_edges_deg = edges
        self.bin_centers_deg = (edges[:-1] + edges[1:]) * 0.5
        self.num_bins = int(len(self.bin_centers_deg))

        self.prob_bins = self.build_bin_probabilities(bin_weights=bin_weights)

    def build_bin_probabilities(
        self, bin_weights: Optional[Sequence[int]]
    ) -> np.ndarray:
        """
        Build the categorical distribution over bins.

        Args:
            bin_weights: Observed counts per bin, used only if strategy="inverse_freq".

        Returns:
            NumPy array of probabilities with shape [num_bins].
        """
        if self.strategy == "inverse_freq" and bin_weights is not None:
            w = np.asarray(bin_weights, dtype=np.float64)
            if w.shape[0] != self.num_bins:
                raise ValueError(
                    f"bin_weights length must be {self.num_bins}, got {w.shape[0]}."
                )
            w = np.maximum(w, 1e-8)
            p = 1.0 / w
            p = p / p.sum()
            return p.astype(np.float64)

        return (np.ones(self.num_bins, dtype=np.float64) / self.num_bins).astype(
            np.float64
        )

    def choose_reference_index(self, boxes: torch.Tensor, policy: str) -> int:
        """
        Choose a GT box index to use as reference for deciding the rotation.

        Args:
            boxes: Tensor [N, 8] with OBB corners.
            policy: "first", "random", or "largest" (largest AABB area of the OBB).

        Returns:
            The selected index in [0, N-1], or -1 if N == 0.
        """
        num = int(boxes.shape[0])
        if num == 0:
            return -1

        if policy == "first":
            return 0
        if policy == "random":
            return int(np.random.randint(0, num))

        pts = boxes.view(num, 4, 2).detach().cpu().numpy()
        widths = pts[..., 0].max(axis=1) - pts[..., 0].min(axis=1)
        heights = pts[..., 1].max(axis=1) - pts[..., 1].min(axis=1)
        areas = widths * heights
        return int(np.argmax(areas))

    def delta_to_bin_center(self, theta_rad: float, center_deg: float) -> float:
        """
        Compute a rotation delta (radians) that moves a reference angle close to a target bin center.

        Because your histogram is built over [0, 180) using |angle|, there are two equivalent
        target directions (+center and -center). We choose the delta that minimizes absolute rotation.

        Args:
            theta_rad: Reference GT angle in radians.
            center_deg: Target bin center in degrees.

        Returns:
            Rotation delta in radians (wrapped to [-pi, pi]) that is smallest in magnitude.
        """
        target = math.radians(center_deg)

        cand1 = wrap_to_pi(theta_rad - target)
        cand2 = wrap_to_pi(theta_rad + target)

        return cand1 if abs(cand1) <= abs(cand2) else cand2

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply the transform to a sample with probability `self.prob`.

        Args:
            sample: A dict containing "image" and "target".

        Returns:
            The (possibly) rotated sample dict.
        """
        if float(np.random.rand()) > self.prob:
            return sample

        image = sample["image"]
        target = sample["target"]

        boxes = target.get("boxes", None)
        angles = target.get("angles", None)

        if boxes is None or angles is None or boxes.numel() == 0:
            random_phi = math.radians(
                float(np.random.uniform(-self.max_angle, self.max_angle))
            )
            return self.apply_rotation(sample=sample, angle_rad=random_phi)

        ref_idx = self.choose_reference_index(boxes=boxes, policy=self.ref_policy)
        if ref_idx < 0:
            random_phi = math.radians(
                float(np.random.uniform(-self.max_angle, self.max_angle))
            )
            return self.apply_rotation(sample=sample, angle_rad=random_phi)

        theta_ref = float(angles[ref_idx].item())
        limit = math.radians(self.max_angle) + 1e-6

        for _ in range(self.max_tries):
            k = int(np.random.choice(self.num_bins, p=self.prob_bins))
            delta = self.delta_to_bin_center(
                theta_rad=theta_ref, center_deg=float(self.bin_centers_deg[k])
            )
            if abs(delta) <= limit:
                return self.apply_rotation(sample=sample, angle_rad=delta)

        random_phi = math.radians(
            float(np.random.uniform(-self.max_angle, self.max_angle))
        )
        return self.apply_rotation(sample=sample, angle_rad=random_phi)

    def apply_rotation(
        self, sample: Dict[str, Any], angle_rad: float
    ) -> Dict[str, Any]:
        """
        Rotate the image and update OBB corners and angles in the target.

        Rotation is applied around the image center, with canvas expansion to fit the rotated image.

        Args:
            sample: Sample dict with keys "image" and "target".
            angle_rad: Rotation angle in radians (positive is CCW in OpenCV degrees convention).

        Returns:
            Updated sample dict with rotated image and updated target.
        """
        image = sample["image"]
        target = sample["target"]

        height, width = image.shape[:2]
        angle_deg = math.degrees(angle_rad)

        cos_a, sin_a = abs(math.cos(angle_rad)), abs(math.sin(angle_rad))
        new_w = int(height * sin_a + width * cos_a)
        new_h = int(height * cos_a + width * sin_a)

        rot = cv2.getRotationMatrix2D(
            (width / 2.0, height / 2.0), angle_deg, 1.0
        ).astype(np.float32)
        rot[0, 2] += (new_w - width) / 2.0
        rot[1, 2] += (new_h - height) / 2.0

        rotated_image = cv2.warpAffine(
            image, rot, (new_w, new_h), flags=cv2.INTER_LINEAR
        )

        boxes = target.get("boxes", None)
        angles = target.get("angles", None)

        num = int(boxes.shape[0]) if isinstance(boxes, torch.Tensor) else 0
        if boxes is not None and isinstance(boxes, torch.Tensor) and num > 0:
            device = boxes.device
            pts = boxes.view(num, 4, 2).detach().cpu().numpy().astype(np.float32)
            hom = np.concatenate([pts, np.ones((num, 4, 1), dtype=np.float32)], axis=2)
            pts_rot = hom @ rot.T  # [N, 4, 2]

            target["boxes"] = torch.tensor(
                pts_rot.reshape(num, 8), dtype=torch.float32, device=device
            )

            if (
                angles is not None
                and isinstance(angles, torch.Tensor)
                and angles.numel() == num
            ):
                target["angles"] = wrap_to_pi(angles - float(angle_rad))

        sample["image"] = rotated_image
        sample["target"] = target

        boxes_after = target.get("boxes", None)
        if isinstance(boxes_after, torch.Tensor) and boxes_after.numel() > 0:
            n_after = int(boxes_after.shape[0])
            sample["target"]["valid_mask"] = torch.ones(
                n_after, dtype=torch.bool, device=boxes_after.device
            )
        else:
            sample["target"]["valid_mask"] = torch.zeros(0, dtype=torch.bool)

        return sample


def collect_degrees_by_class(
    dataset: Any,
    labels_map: Dict[int, str],
) -> Dict[str, Any]:
    """
    Iterate over a dataset and collect GT angles (in degrees) for all samples and per class.

    The dataset is expected to return a dict sample with:
      sample["target"]["angles"]     Tensor [N]
      sample["target"]["class_idx"]  Tensor [N]
      sample["target"]["valid_mask"] Optional[Tensor [N]] boolean

    Args:
        dataset: Any indexable dataset with __len__ and __getitem__.
        labels_map: Mapping class_idx -> class name.

    Returns:
        Dict with:
            - "all": List[float] of degrees for all valid GTs
            - "per_cls": Dict[int, List[float]] of degrees per class
            - "counts": Dict[int, int] number of entries per class
    """
    per_cls: Dict[int, List[float]] = {c: [] for c in labels_map.keys()}
    all_deg: List[float] = []

    for i in range(len(dataset)):
        sample = dataset[i]
        target = sample["target"]

        angles = target["angles"]
        classes = target["class_idx"]

        if angles.numel() == 0:
            continue

        valid_mask = target.get(
            "valid_mask", torch.ones_like(classes, dtype=torch.bool)
        )
        angles = angles[valid_mask]
        classes = classes[valid_mask]

        deg = angles_rad_to_deg_0_180(angles)
        all_deg.extend([float(x) for x in deg.tolist()])

        for d, c in zip(deg.tolist(), classes.tolist()):
            if int(c) in per_cls:
                per_cls[int(c)].append(float(d))

    counts = {c: len(per_cls[c]) for c in labels_map.keys()}
    return {"all": all_deg, "per_cls": per_cls, "counts": counts}


def build_bin_weights_from_degrees(all_deg: List[float], bin_deg: int) -> List[int]:
    """
    Build histogram bin counts over [0, 180] degrees.

    Args:
        all_deg: List of degrees values in [0, 180).
        bin_deg: Bin width in degrees.

    Returns:
        List[int] with length 180/bin_deg containing counts per bin.
    """
    bins = np.arange(0, 180 + bin_deg, bin_deg)
    counts, _ = np.histogram(all_deg, bins=bins)
    return counts.tolist()


def save_counts_csv(
    path: Path,
    stats: Dict[str, Any],
    bin_deg: int,
    labels_map: Dict[int, str],
) -> None:
    """
    Save histogram bin counts to CSV for all classes and per-class breakdown.

    Output rows contain:
      scope, class_idx, class_name, bin_left, bin_right, count

    Args:
        path: Output CSV path.
        stats: Output of `collect_degrees_by_class`.
        bin_deg: Bin width in degrees.
        labels_map: Mapping class_idx -> class name.
    """
    bins = np.arange(0, 180 + bin_deg, bin_deg)
    rows: List[Dict[str, Any]] = []

    counts_all, edges = np.histogram(stats["all"], bins=bins)
    for i, cnt in enumerate(counts_all):
        rows.append(
            {
                "scope": "ALL",
                "class_idx": "ALL",
                "class_name": "ALL",
                "bin_left": int(edges[i]),
                "bin_right": int(edges[i + 1]),
                "count": int(cnt),
            }
        )

    for class_idx, class_name in labels_map.items():
        counts_c, _ = np.histogram(stats["per_cls"][class_idx], bins=bins)
        for i, cnt in enumerate(counts_c):
            rows.append(
                {
                    "scope": "PERCLASS",
                    "class_idx": int(class_idx),
                    "class_name": str(class_name),
                    "bin_left": int(edges[i]),
                    "bin_right": int(edges[i + 1]),
                    "count": int(cnt),
                }
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
