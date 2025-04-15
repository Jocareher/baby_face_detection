import cv2
import math
import random
import numpy as np
import torch
from typing import Tuple


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
    - class indices (0↔1, 3↔4)
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
                "class_idxs"
            ].clone()  # Creates a copy of the class indices tensor.

            # Flip X coordinates
            boxes = boxes.view(-1, 4, 2)  # (N, 4, 2)
            boxes[..., 0] = w - boxes[..., 0]  # Flip x-coordinates

            # Reorder points to maintain orientation: [1, 0, 3, 2]
            reorder_idx = torch.tensor([1, 0, 3, 2], device=boxes.device)
            boxes = boxes[:, reorder_idx, :]
            boxes = boxes.view(-1, 8)  # Back to (N, 8)

            # Negate angles
            angles = -angles

            # Flip class indices: vectorized swap using masks
            class_idxs_flipped = class_idxs.clone()
            swap_map = {0: 1, 1: 0, 3: 4, 4: 3}
            for a, b in swap_map.items():
                class_idxs_flipped[class_idxs == a] = b

            target["boxes"] = boxes  # Updates the boxes in the target dictionary.
            target["angles"] = angles  # Updates the angles in the target dictionary.
            target[
                "class_idxs"
            ] = class_idxs_flipped  # Updates the class indices in the target dictionary.

        sample["image"] = image  # Updates the image in the sample.
        sample["target"] = target  # Updates the target in the sample.
        return sample


class RandomRotateOBB:
    """
    Randomly rotates the image and OBBs by an angle in degrees between [-max_angle, max_angle],
    expanding the canvas to avoid cropping, and normalizing the resulting angles.
    """

    def __init__(self, max_angle: int = 180, prob: float = 0.5):
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
        angle_deg = -random.uniform(
            -self.max_angle, self.max_angle
        )  # clockwise. Generates a random rotation angle.
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
            "class_idxs"
        ].clone()  # Creates a copy of the class indices tensor.

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

        # Update angle and normalize to [0, 2π)
        angles = (angles + angle_rad) % (2 * math.pi)

        target["boxes"] = boxes  # Updates the boxes in the target dictionary.
        target["angles"] = angles  # Updates the angles in the target dictionary.
        target[
            "class_idxs"
        ] = class_idxs  # Updates the class indices in the target dictionary.

        sample["image"] = rotated_image  # Updates the rotated image in the sample.
        sample["target"] = target  # Updates the target in the sample.
        return sample


def normalize_angle(angle_rad: float) -> float:
    """
    Normalizes an angle in radians to the range [0, 2π).
    """
    return angle_rad % (2 * math.pi)  # Calcula el ángulo normalizado.


class RandomScaleTranslateOBB:
    """
    Randomly scales and translates the image and its OBBs.
    Canvas is expanded to avoid cropping. OBBs completely outside the frame are removed.
    """

    def __init__(
        self,
        scale_range: Tuple[float, float] = (0.9, 1.1),
        translate_range: Tuple[float, float] = (-0.2, 0.2),
        prob: float = 0.5,
    ):
        """
        Initializes the RandomScaleTranslateOBB transform.

        Args:
            scale_range (Tuple[float, float]): The range of scaling factors. Defaults to (0.9, 1.1).
            translate_range (Tuple[float, float]): The range of translation factors. Defaults to (-0.2, 0.2).
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
        h, w = image.shape[:2]

        # Randomly generates scale and translation factors.
        scale = random.uniform(*self.scale_range)
        # Calculates the translation factors based on the image dimensions.
        tx = random.uniform(*self.translate_range) * w
        ty = random.uniform(*self.translate_range) * h

        # Calculates the new width and height of the canvas.
        new_w = int(w * scale + abs(tx))
        new_h = int(h * scale + abs(ty))

        # Calculates the translation matrix.
        M = np.array(
            [[scale, 0, tx if tx > 0 else -tx], [0, scale, ty if ty > 0 else -ty]],
            dtype=np.float32,
        )

        # Applies the translation to the rotation matrix.
        transformed_image = cv2.warpAffine(
            image, M, (new_w, new_h), flags=cv2.INTER_LINEAR
        )

        # Adjusts the translation matrix for the new canvas size.
        boxes = target["boxes"]
        angles = target["angles"]
        class_idxs = target["class_idxs"]

        # Vectorized transform
        if boxes.shape[0] == 0:
            # No boxes to transform
            target["boxes"] = torch.empty((0, 8), dtype=torch.float32)
            target["angles"] = torch.empty((0,), dtype=torch.float32)
            target["class_idxs"] = torch.tensor([5], dtype=torch.long)
        else:
            # Vectorized transform
            N = boxes.shape[0]
            boxes_np = boxes.view(N, 4, 2).cpu().numpy()  # (N, 4, 2)
            ones = np.ones((N, 4, 1), dtype=np.float32)
            boxes_hom = np.concatenate([boxes_np, ones], axis=2)  # (N, 4, 3)
            transformed_boxes = np.matmul(boxes_hom, M.T)  # (N, 4, 2)

            # Check validity mask (all 4 points must lie inside the new canvas)
            x_in_bounds = (0 <= transformed_boxes[:, :, 0]) & (
                transformed_boxes[:, :, 0] < new_w
            )
            y_in_bounds = (0 <= transformed_boxes[:, :, 1]) & (
                transformed_boxes[:, :, 1] < new_h
            )
            is_valid = (x_in_bounds & y_in_bounds).all(axis=1)  # (N,)

            # Reshape to (N, 8) and filter out invalid boxes
            valid_boxes = transformed_boxes[is_valid].reshape(-1, 8)
            valid_angles = angles[is_valid]
            valid_class_idxs = class_idxs[is_valid]

            # If no boxes are valid, assign empty tensors
            if valid_boxes.shape[0] == 0:
                target["boxes"] = torch.empty((0, 8), dtype=torch.float32)
                target["angles"] = torch.empty((0,), dtype=torch.float32)
                target["class_idxs"] = torch.tensor([5], dtype=torch.long)
            else:
                target["boxes"] = torch.tensor(
                    valid_boxes, dtype=torch.float32, device=boxes.device
                )
                target["angles"] = valid_angles
                target["class_idxs"] = valid_class_idxs

        # Updates the target in the sample.
        sample["image"] = transformed_image
        sample["target"] = target
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
        prob: float = 0.5,
    ):
        """
        Initializes the ColorJitterOBB transform.

        Args:
            brightness (float): The brightness adjustment factor. Defaults to 0.2.
            contrast (float): The contrast adjustment factor. Defaults to 0.2.
            saturation (float): The saturation adjustment factor. Defaults to 0.2.
            prob (float): The probability of applying the transform. Defaults to 0.5.
        """
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
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
