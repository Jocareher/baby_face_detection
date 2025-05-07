import math
import random
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Polygon as MplPolygon


from loss.utils import xyxyxyxy2xywhr, xywhr2xyxyxyxy, decode_vertices


def denormalize_image(
    img_tensor: torch.Tensor,
    mean: Tuple[float, float, float] = (0.6427, 0.5918, 0.5526),
    std: Tuple[float, float, float] = (0.2812, 0.2825, 0.3036),
) -> np.ndarray:
    """
    Converts a normalized image tensor (C x H x W) back to a NumPy array (H x W x C) in uint8 format.

    Args:
        img_tensor (Tensor): The normalized image tensor.
        mean (tuple): Mean used during normalization (per channel).
        std (tuple): Std used during normalization (per channel).

    Returns:
        np.ndarray: Denormalized image in uint8 format.
    """
    img_np = img_tensor.cpu().numpy()  # Convert the tensor to a NumPy array on CPU.
    for c in range(3):  # Iterate through each color channel.
        img_np[c] = img_np[c] * std[c] + mean[c]  # Denormalize the channel.
    img_np = np.clip(img_np * 255.0, 0, 255).astype(
        np.uint8
    )  # Clip and convert to uint8.
    return img_np.transpose(1, 2, 0)  # Convert to H x W x C format.


def draw_obb(
    ax,
    box,
    angle: Optional[float] = None,
    class_idx: Optional[int] = None,
    top_color: str = "red",
    other_color: str = "blue",
    linewidth: int = 2,
):
    """
    Draws an oriented bounding box (OBB) and annotates it with class index and angle.

    Args:
        ax: Matplotlib axis.
        box: List or array of 8 values [x1, y1, ..., x4, y4].
        angle: Rotation angle in radians (optional).
        class_idx: Integer class index (optional).
        top_color: Color of the top edge of the OBB.
        other_color: Color of the other edges of the OBB.
        linewidth: Line width for the OBB.
    """
    pts = np.array(box).reshape(4, 2)  # Reshape the box coordinates to (4, 2).
    pts_closed = np.vstack(
        [pts, pts[0]]
    )  # Close the polygon by adding the first point again.
    ax.plot(
        pts_closed[:, 0], pts_closed[:, 1], color=other_color, linewidth=linewidth
    )  # Plot the OBB edges.
    ax.plot(
        [pts[0, 0], pts[1, 0]],
        [pts[0, 1], pts[1, 1]],
        color=top_color,
        linewidth=linewidth + 1,
    )  # Highlight the top edge.

    # Class label near (x1, y1)
    if class_idx is not None:
        ax.text(
            pts[0, 0],
            pts[0, 1] - 5,
            f"cls: {class_idx}",
            color="green",
            fontsize=10,
            weight="bold",
        )  # Add class label.

    # Angle annotation at center of the box
    if angle is not None:
        center = pts.mean(axis=0)  # Calculate the center of the OBB.
        angle_deg = np.degrees(angle)  # Convert angle to degrees.
        ax.text(
            center[0],
            center[1],
            f"{angle_deg:.1f}°",
            color="orange",
            fontsize=9,
            ha="center",
            va="center",
        )  # Add angle annotation.


def visualize_dataset(dataset, num_images: int = 9):
    """
    Displays 'num_images' samples from the dataset in a grid.
    Shows OBBs with segment highlighting, class_idx, angle, and image filename.
    """
    total = len(dataset)  # Get the total number of samples in the dataset.
    if total == 0:  # Check if the dataset is empty.
        print("Dataset is empty.")
        return

    indices = random.sample(
        range(total), min(num_images, total)
    )  # Select random indices.
    cols = int(
        math.ceil(math.sqrt(len(indices)))
    )  # Calculate the number of columns for the grid.
    rows = int(
        math.ceil(len(indices) / cols)
    )  # Calculate the number of rows for the grid.

    fig, axes = plt.subplots(
        rows, cols, figsize=(cols * 5, rows * 5)
    )  # Create the figure and axes.
    axes = np.array(axes).reshape(-1)  # Reshape the axes array to a 1D array.

    for ax in axes[len(indices) :]:  # Turn off axes for empty subplots.
        ax.axis("off")

    for i, idx in enumerate(indices):  # Iterate through the selected indices.
        sample = dataset[idx]  # Get the sample.
        image = sample["image"]  # Get the image.
        if torch.is_tensor(image):  # Check if the image is a tensor.
            image_np = denormalize_image(image)  # Denormalize the image.
        else:
            image_np = image.copy()  # Create a copy of the image.

        ax = axes[i]  # Get the current axis.
        ax.imshow(image_np)  # Display the image.
        ax.axis("off")  # Turn off the axis.

        boxes = sample["target"]["boxes"]  # Get the bounding boxes.
        angles = sample["target"]["angles"]  # Get the angles.
        class_idxs = sample["target"]["class_idx"]  # Get the class indices.

        if torch.is_tensor(boxes):  # Check if the boxes are tensors.
            boxes = boxes.cpu().numpy()  # Convert the boxes to NumPy arrays.
        if torch.is_tensor(angles):  # Check if the angles are tensors.
            angles = angles.cpu().numpy()  # Convert the angles to NumPy arrays.
        if torch.is_tensor(class_idxs):  # Check if the class indices are tensors.
            class_idxs = (
                class_idxs.cpu().numpy()
            )  # Convert the class indices to NumPy arrays.

        for j in range(len(boxes)):  # Iterate through the bounding boxes.
            draw_obb(
                ax,
                box=boxes[j],
                angle=angles[j] if j < len(angles) else None,
                class_idx=class_idxs[j] if j < len(class_idxs) else None,
                top_color="red",
                other_color="blue",
                linewidth=2,
            )  # Draw the OBB.

        # Add filename title
        base_name = dataset.file_list[idx]  # Get the filename.
        ax.set_title(f"{base_name}.jpg", fontsize=11, color="black")  # Set the title.

    plt.tight_layout()  # Adjust the subplot parameters to give specified padding.
    plt.show()  # Display the plot.


def visualize_predictions(
    images, pred_obbs, pred_angles, gt_obbs, gt_angles, anchors, image_sizes
):
    """
    Optionally visualizes the predicted vs ground truth oriented bounding boxes (OBBs)
    for the second image in the batch. Intended for debugging or qualitative inspection.

    Args:
        images (Tensor): Batch of input images (B, C, H, W).
        pred_obbs (Tensor): Predicted offset vertices, shape (B, N, 8).
        pred_angles (Tensor): Predicted angles, shape (B, N, 1).
        gt_obbs (Tensor): Ground truth OBBs, shape (B, N, 8).
        gt_angles (Tensor): Ground truth angles, shape (B, N, 1).
        anchors (Tensor): Anchor vertices in pixel space, shape (B, N, 8).
        image_sizes (List[Tuple[int, int]]): List of image sizes in (W, H) format.
    """
    W, H = image_sizes[0]

    pred_xy = decode_vertices(pred_obbs[0], anchors[0], (W, H))
    pred_xywhr = xyxyxyxy2xywhr(pred_xy, pred_angles[0].squeeze(-1), (W, H))
    gt_xywhr = xyxyxyxy2xywhr(gt_obbs[0], gt_angles[1], (W, H))

    print("Pred:", pred_xywhr[0].tolist())
    print("GT:", gt_xywhr[0].tolist())

    show_obbs_on_image(
        images[0], pred_xywhr[0].unsqueeze(0), gt_xywhr[0].unsqueeze(0), (W, H)
    )

    def show_obbs_on_image(
        image_tensor: torch.Tensor,
        pred_xywhr: torch.Tensor,
        gt_xywhr: torch.Tensor,
        image_size: Tuple[int, int],
    ):
        """
        Draws predicted and ground truth OBBs over the original image.

        Args:
            image_tensor (Tensor): Normalized image tensor (3, H, W).
            pred_xywhr (Tensor): Predicted OBBs (N, 5) in [cx, cy, w, h, angle] format.
            gt_xywhr (Tensor): Ground truth OBBs (N, 5) in [cx, cy, w, h, angle] format.
            image_size (Tuple[int, int]): (W, H) dimensions of the image.

        """

        # Convert image tensor to numpy array
        image_np = denormalize_image(image_tensor)
        # Convert to 4 corner format
        pred_corners = xywhr2xyxyxyxy(pred_xywhr)
        gt_corners = xywhr2xyxyxyxy(gt_xywhr)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(image_np)
        ax.set_title("OBB Prediction vs Ground Truth")
        ax.axis("off")

        # Draw GT
        for i in range(len(gt_corners)):
            # Draw the GT corners
            # Note: gt_corners[i] is in shape (4, 2)
            # and gt_xywhr[i] is in shape (5,)
            ax.add_patch(
                MplPolygon(
                    gt_corners[i],
                    closed=True,
                    fill=False,
                    edgecolor="blue",
                    linewidth=1.5,
                    label="GT" if i == 0 else None,
                )
            )
            ax.scatter(*gt_xywhr[i, :2].detach().cpu().numpy(), color="blue", s=10)

        # Draw predictions
        for i in range(len(pred_corners)):
            # Draw the predicted corners
            # Note: pred_corners[i] is in shape (4, 2)
            # and pred_xywhr[i] is in shape (5,)
            ax.add_patch(
                MplPolygon(
                    pred_corners[i],
                    closed=True,
                    fill=False,
                    edgecolor="red",
                    linestyle="--",
                    linewidth=1.5,
                    label="Pred" if i == 0 else None,
                )
            )
            ax.scatter(
                *pred_xywhr[i, :2].detach().cpu().numpy(), color="red", marker="x", s=10
            )

        # Only show unique legend labels
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc="lower right")

        plt.tight_layout()
        plt.show()
