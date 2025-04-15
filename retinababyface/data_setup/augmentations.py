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
        for i in range(boxes.shape[0]):  # Iterates through each bounding box.
            box = boxes[i]  # Gets the i-th bounding box.
            box[0] *= scale_x  # Scales the x1 coordinate.
            box[2] *= scale_x  # Scales the x2 coordinate.
            box[4] *= scale_x  # Scales the x3 coordinate.
            box[6] *= scale_x  # Scales the x4 coordinate.
            box[1] *= scale_y  # Scales the y1 coordinate.
            box[3] *= scale_y  # Scales the y2 coordinate.
            box[5] *= scale_y  # Scales the y3 coordinate.
            box[7] *= scale_y  # Scales the y4 coordinate.
            boxes[i] = box  # Updates the bounding box in the tensor.

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

            for i in range(boxes.shape[0]):  # Iterates through each bounding box.
                box = boxes[i].view(4, 2)  # Reshapes the bounding box to (4, 2).
                box[:, 0] = w - box[:, 0]  # Flips the X coordinates.
                box = box[
                    [1, 0, 3, 2]
                ]  # Reorders the coordinates to keep the OBB orientation.
                boxes[i] = box.view(-1)  # Reshapes the bounding box back to (8,).

                angles[i] = -angles[i]  # Negates the angle.

                # Flip class indices
                if class_idxs[i] == 0:  # Flips class index 0 to 1.
                    class_idxs[i] = 1
                elif class_idxs[i] == 1:  # Flips class index 1 to 0.
                    class_idxs[i] = 0
                elif class_idxs[i] == 3:  # Flips class index 3 to 4.
                    class_idxs[i] = 4
                elif class_idxs[i] == 4:  # Flips class index 4 to 3.
                    class_idxs[i] = 3

            target["boxes"] = boxes  # Updates the boxes in the target dictionary.
            target["angles"] = angles  # Updates the angles in the target dictionary.
            target[
                "class_idxs"
            ] = class_idxs  # Updates the class indices in the target dictionary.

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

        for i in range(boxes.shape[0]):  # Iterates through each bounding box.
            box = (
                boxes[i].view(4, 2).numpy()
            )  # Reshapes the bounding box to (4, 2) and converts it to a NumPy array.
            ones = np.ones(
                (4, 1)
            )  # Creates a tensor of ones for homogeneous coordinates.
            box_hom = np.hstack(
                [box, ones]
            )  # Concatenates the bounding box with the ones.
            rotated_box = np.dot(rot_mat, box_hom.T).T  # Rotates the bounding box.
            boxes[i] = torch.tensor(
                rotated_box.flatten(), dtype=torch.float32
            )  # Updates the bounding box in the tensor.

            # Update angle and normalize to [0, 2π)
            new_angle = normalize_angle(
                angles[i].item() + angle_rad
            )  # Calculates the new angle and normalizes it.
            angles[i] = new_angle  # Updates the angle in the tensor.

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
        if random.random() > self.prob:  # Checks if the transform should be applied.
            return sample

        image, target = (
            sample["image"],
            sample["target"],
        )  # Extracts the image and target from the sample.
        h, w = image.shape[:2]  # Gets the height and width of the image.

        # Sample scale factor and translations
        scale = random.uniform(*self.scale_range)  # Generates a random scaling factor.
        tx = (
            random.uniform(*self.translate_range) * w
        )  # Generates a random translation in the x-axis.
        ty = (
            random.uniform(*self.translate_range) * h
        )  # Generates a random translation in the y-axis.

        # New canvas size to accommodate transformed image
        new_w = int(w * scale + abs(tx))  # Calculates the new width of the canvas.
        new_h = int(h * scale + abs(ty))  # Calculates the new height of the canvas.

        # Build affine matrix
        M = np.array(
            [[scale, 0, tx if tx > 0 else -tx], [0, scale, ty if ty > 0 else -ty]],
            dtype=np.float32,
        )  # Creates the affine transformation matrix.

        # Transform image
        transformed_image = cv2.warpAffine(
            image, M, (new_w, new_h), flags=cv2.INTER_LINEAR
        )  # Transforms the image.

        boxes = target["boxes"].clone()  # Creates a copy of the bounding boxes tensor.
        angles = target["angles"].clone()  # Creates a copy of the angles tensor.
        class_idxs = target[
            "class_idxs"
        ].clone()  # Creates a copy of the class indices tensor.

        valid_boxes = []  # List to store valid bounding boxes.
        valid_angles = []  # List to store valid angles.
        valid_class_idxs = []  # List to store valid class indices.

        for i in range(boxes.shape[0]):  # Iterates through each bounding box.
            box = (
                boxes[i].view(4, 2).numpy()
            )  # Reshapes the bounding box to (4, 2) and converts it to a NumPy array.
            ones = np.ones(
                (4, 1)
            )  # Creates a tensor of ones for homogeneous coordinates.
            box_hom = np.hstack(
                [box, ones]
            )  # Concatenates the bounding box with the ones.
            transformed_box = np.dot(M, box_hom.T).T  # Transforms the bounding box.

            # Check if all points are within the new image
            if np.all(
                (0 <= transformed_box[:, 0])
                & (transformed_box[:, 0] < new_w)
                & (0 <= transformed_box[:, 1])
                & (transformed_box[:, 1] < new_h)
            ):  # Checks if the bounding box is within the new image.
                valid_boxes.append(
                    transformed_box.flatten()
                )  # Adds the transformed bounding box to the list.
                valid_angles.append(
                    angles[i].item()
                )  # angle doesn't change. Adds the angle to the list.
                valid_class_idxs.append(
                    class_idxs[i].item()
                )  # Adds the class index to the list.

        # Build target
        if len(valid_boxes) == 0:  # Checks if there are any valid bounding boxes.
            target["boxes"] = torch.empty(
                (0, 8), dtype=torch.float32
            )  # Creates an empty tensor for boxes.
            target["angles"] = torch.empty(
                (0,), dtype=torch.float32
            )  # Creates an empty tensor for angles.
            target["class_idxs"] = torch.tensor(
                [5], dtype=torch.long
            )  # background. Creates a tensor for background class index.
        else:
            target["boxes"] = torch.tensor(
                np.array(valid_boxes), dtype=torch.float32
            )  # Creates a tensor for valid boxes.
            target["angles"] = torch.tensor(
                valid_angles, dtype=torch.float32
            )  # Creates a tensor for valid angles.
            target["class_idxs"] = torch.tensor(
                valid_class_idxs, dtype=torch.long
            )  # Creates a tensor for valid class indices.

        sample[
            "image"
        ] = transformed_image  # Updates the transformed image in the sample.
        sample["target"] = target  # Updates the target in the sample.
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
            mean = image.mean(
                axis=(0, 1), keepdims=True
            )  # Calculates the mean of the image.
            factor = 1.0 + random.uniform(
                -self.contrast, self.contrast
            )  # Generates a random contrast factor.
            image = (image - mean) * factor + mean  # Adjusts the contrast of the image.

        # Saturation (only affects RGB, so convert to HSV)
        if self.saturation > 0:  # Checks if saturation adjustment should be applied.
            hsv = cv2.cvtColor(
                np.clip(image, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV
            ).astype(
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

        sample["image"] = np.clip(image, 0, 255).astype(
            np.uint8
        )  # Clips the image values to [0, 255] and converts it to uint8.
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
        noise = np.random.normal(0, self.std, image.shape).astype(
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
        blurred = cv2.GaussianBlur(
            image, self.ksize, 0
        )  # Applies Gaussian blur to the image.
        sample["image"] = blurred  # Updates the blurred image in the sample.
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
            max_x = int(
                max(x_min, 0)
            )  # Gets the maximum x coordinate for the occlusion.
            max_y = int(
                max(y_min, 0)
            )  # Gets the maximum y coordinate for the occlusion.
            range_x = max(
                int(x_max - occ_w), max_x + 1
            )  # Calculates the range for the x coordinate of the occlusion.
            range_y = max(
                int(y_max - occ_h), max_y + 1
            )  # Calculates the range for the y coordinate of the occlusion.

            x0 = random.randint(
                max_x, max(range_x, max_x + 1)
            )  # Generates a random x coordinate for the occlusion.
            y0 = random.randint(
                max_y, max(range_y, max_y + 1)
            )  # Generates a random y coordinate for the occlusion.
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
            torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        )  # Converts the image to a PyTorch tensor and normalizes it.
        image = (
            image - self.mean
        ) / self.std  # Normalizes the image using the mean and standard deviation.
        sample["image"] = image  # Updates the normalized image in the sample.
        return sample
