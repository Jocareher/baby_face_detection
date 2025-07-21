import math
import random
import re
import os
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Polygon
from PIL import Image, ImageDraw, ImageFont


from loss.utils import xyxyxyxy2xywhr, xywhr2xyxyxyxy, decode_vertices


def denormalize_image(
    img_tensor: torch.Tensor,
    mean=(0.6427, 0.5918, 0.5525),
    std=(0.2812, 0.2825, 0.3036),
) -> np.ndarray:
    """
    Reverts normalization on an image tensor and converts it to a NumPy array for visualization.

    Args:
        img_tensor (torch.Tensor): Normalized image tensor of shape (C, H, W), with values in [-1, 1] or [0, 1].
        mean (Tuple[float, float, float]): Mean values per channel used during normalization.
        std (Tuple[float, float, float]): Std deviation per channel used during normalization.

    Returns:
        np.ndarray: Denormalized image in (H, W, C) format, dtype=uint8, with pixel values in [0, 255].
    """
    img = (
        img_tensor.clone().detach().cpu()
    )  # Ensure we don't modify the original tensor
    for t, m, s in zip(img, mean, std):  # Apply inverse normalization per channel
        t.mul_(s).add_(m)
    img = torch.clamp(img, 0, 1)  # Clamp to valid pixel range [0, 1]
    img = (img * 255).byte().numpy()  # Convert to uint8 [0, 255]
    return np.transpose(img, (1, 2, 0))  # Rearrange to H x W x C for image display


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


def visualize_dataset(dataset, num_images: int = 9, show: bool = False):
    """
    Displays 'num_images' samples from the dataset in a grid.
    Shows OBBs with segment highlighting, class_idx, angle, and image filename.
    Args:
        dataset (Dataset): PyTorch dataset with 'image' and 'target' keys.
        num_images (int): Number of images to display.
        show (bool): Whether to display the plot or not.
    Returns:
        fig (Figure): Matplotlib figure object.
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
    if show:
        plt.show()
    return fig


def visualize_and_save_dataset_in_script(
    dataset, split_name: str, save_dir: str, num_images: int = 9
):
    """
    Visualizes a sample of the dataset and saves the result as a grid image.

    Args:
        dataset (Dataset): PyTorch dataset with 'image' and 'target' keys.
        split_name (str): Name of the dataset split (e.g., 'train', 'val', 'test').
        save_dir (str): Path to directory where image will be saved.
        num_images (int): Number of images to display.
    """
    fig = visualize_dataset(dataset, num_images=num_images)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{split_name}_grid.png")
    if fig:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[INFO] Saved {split_name} visualization grid to {save_path}")


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
                Polygon(
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
                Polygon(
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


def create_training_gif(
    image_folder: str, output_gif: str, font_path: Optional[str] = None
):
    """
    Creates a training progress GIF with large bold titles indicating the epoch number.

    Args:
        image_folder (str): Folder with .jpg/.png images named with 'epoch{num}'.
        output_gif (str): Path where the animated GIF will be saved.
        font_path (Optional[str]): Optional path to a .ttf font to use.
    """
    valid_exts = [".jpg", ".png"]
    image_files = [
        f
        for f in os.listdir(image_folder)
        if os.path.splitext(f)[1].lower() in valid_exts
    ]

    def extract_epoch(filename: str) -> int:
        match = re.search(r"epoch(\d+)", filename, re.IGNORECASE)
        return int(match.group(1)) if match else -1

    images = [(extract_epoch(f), f) for f in image_files if extract_epoch(f) >= 0]
    images.sort(key=lambda x: x[0])

    if not images:
        print("[WARNING] No valid images with epoch numbers found.")
        return

    frames = []

    for epoch, filename in images:
        img_path = os.path.join(image_folder, filename)
        img = Image.open(img_path).convert("RGB")
        width, height = img.size

        # --- Reserve space for title banner ---
        title_height = int(height * 0.1)  # 12% of image height
        canvas = Image.new("RGB", (width, height + title_height), color="white")
        canvas.paste(img, (0, title_height))

        draw = ImageDraw.Draw(canvas)
        font_size = int(title_height * 0.3)

        # --- Font loading ---
        # Font loading with robust fallback
        try:
            if font_path and os.path.exists(font_path):
                font = ImageFont.truetype(font_path, font_size)
            else:
                try:
                    font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
                except:
                    try:
                        font = ImageFont.truetype(
                            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                            font_size,
                        )
                    except:
                        raise RuntimeError(
                            "[ERROR] No valid TTF font found. Please provide --font_path."
                        )
        except Exception as e:
            raise RuntimeError(f"[ERROR] Font loading failed: {e}")

        # --- Text and placement ---
        text = f"EPOCH : {epoch}"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        text_x = (width - text_width) // 2
        text_y = (title_height - text_height) // 2

        # --- Draw shadow + text ---
        draw.text((text_x + 2, text_y + 2), text, font=font, fill="black")
        draw.text((text_x, text_y), text, font=font, fill="black")

        frames.append(canvas)

    # --- Save GIF ---
    frames[0].save(
        output_gif,
        format="GIF",
        append_images=frames[1:],
        save_all=True,
        duration=500,
        loop=0,
        optimize=True,
    )

    print(f"[INFO] GIF saved to {output_gif} ({len(frames)} frames).")


def visualize_widerface_grid(
    dataset_root: str, rows: int = 3, cols: int = 3, figsize=(15, 10)
) -> None:
    """
    Displays a grid of images with annotated bounding boxes (OBBs) from WIDERFACE dataset.

    Args:
        dataset_root (str): Path to the dataset containing 'images' and 'labels' directories.
        rows (int): Number of rows in the grid. Default is 3.
        cols (int): Number of columns in the grid. Default is 3.
        figsize (tuple): Size of the matplotlib figure. Default is (15, 10).

    The function randomly selects images from the dataset and overlays the bounding boxes
    (oriented bounding boxes) on the images based on the corresponding label files.
    Each bounding box is drawn as a polygon with red edges.
    """
    # Define paths to the images and labels directories
    image_dir = os.path.join(dataset_root, "images")
    label_dir = os.path.join(dataset_root, "labels")

    # Get a list of all image files in the images directory
    image_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]

    # Check if there are any images in the directory
    if len(image_files) == 0:
        print("❌ No images found in the dataset.")
        return

    # Determine the number of images to display in the grid
    num_images = rows * cols
    selected_files = random.sample(image_files, min(num_images, len(image_files)))

    # Create a matplotlib figure with subplots
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()  # Flatten the axes array for easy iteration

    # Iterate over the selected image files and corresponding axes
    for ax, image_file in zip(axes, selected_files):
        img_path = os.path.join(image_dir, image_file)
        label_path = os.path.join(label_dir, os.path.splitext(image_file)[0] + ".txt")

        try:
            # Open the image and display it on the subplot
            img = Image.open(img_path).convert("RGB")
            ax.imshow(img)
        except Exception as e:
            print(f"Error opening {img_path}: {e}")
            continue

        # Get the dimensions of the image
        width, height = img.size

        # Check if the corresponding label file exists
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                lines = f.readlines()

            # Iterate over each line in the label file
            for line in lines:
                parts = line.strip().split()
                # Skip lines that do not have the expected number of elements
                if len(parts) != 11:
                    continue

                # Extract normalized coordinates from the label file
                coords = list(map(float, parts[2:10]))
                # Convert normalized coordinates to absolute pixel positions
                abs_coords = [
                    (coords[i] * width if i % 2 == 0 else coords[i] * height)
                    for i in range(8)
                ]
                # Create a list of (x, y) points for the polygon
                polygon_points = [
                    (abs_coords[i], abs_coords[i + 1]) for i in range(0, 8, 2)
                ]

                # Draw the polygon on the image
                polygon = Polygon(
                    polygon_points, edgecolor="red", facecolor="none", linewidth=2
                )
                ax.add_patch(polygon)

        # Set the title of the subplot to the image file name
        ax.set_title(image_file, fontsize=8)
        ax.axis("off")  # Hide axes for a cleaner display

    # Hide any unused subplots if there are fewer images than grid slots
    for i in range(len(selected_files), len(axes)):
        axes[i].axis("off")

    # Adjust layout to avoid overlapping elements
    plt.tight_layout()
    plt.show()
