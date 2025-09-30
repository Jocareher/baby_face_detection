import math
import random
import re
import os
from pathlib import Path
from typing import Optional, Tuple, Dict

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
    labels_map: Optional[Dict[int, str]] = None,
    edge_color: str = "#008000",  # green
    top_edge: str = "orange",
    linewidth: int = 2,
):
    """
    Draws an Oriented Bounding Box (OBB) with custom styling.

    Args:
        ax: Matplotlib axis to draw on
        box: Array-like of 8 coordinates representing 4 corner points (x,y)
        angle: Optional rotation angle in radians
        class_idx: Optional class index for labeling
        labels_map: Optional dict mapping class indices to readable names
        edge_color: Color for the OBB outline
        top_edge: Color for the diagonal line
        linewidth: Width of drawn lines

    The OBB is drawn with:
    - Dashed green outline
    - Orange diagonal from first to second point
    - Text label showing class name and angle (if provided)
    """
    pts = np.asarray(box, dtype=float).reshape(4, 2)

    # Draw OBB outline
    ax.add_patch(
        Polygon(
            pts,
            closed=True,
            fill=False,
            edgecolor=edge_color,
            linestyle="--",
            linewidth=linewidth,
        )
    )
    # Draw diagonal between first two points
    ax.plot(pts[[0, 1], 0], pts[[0, 1], 1], color=top_edge, linewidth=linewidth)

    # Add class:angle text label
    br_x, br_y = pts[:, 0].max(), pts[:, 1].max()
    cls_txt = (
        labels_map.get(int(class_idx), str(class_idx))
        if (labels_map and class_idx is not None)
        else str(class_idx)
        if class_idx is not None
        else "?"
    )
    ang_txt = f"{math.degrees(float(angle)):.1f}°" if angle is not None else ""
    ax.text(
        br_x,
        br_y,
        f"{cls_txt}: {ang_txt}".strip(": "),
        color="white",
        fontsize=6,
        fontweight="bold",
        ha="right",
        va="bottom",
        bbox=dict(facecolor=edge_color, alpha=0.8, edgecolor="none", pad=2.5),
    )


def visualize_dataset(
    dataset,
    num_images: int = 9,
    labels_map: Optional[Dict[int, str]] = None,
    show: bool = False,
):
    """
    Creates a grid visualization of dataset samples with ground truth annotations.

    Args:
        dataset: PyTorch dataset containing image and target samples
        num_images: Number of random samples to display (default: 9)
        labels_map: Optional mapping from class indices to readable names
        show: Whether to display the plot immediately

    Returns:
        matplotlib.figure.Figure: The generated figure, or None if dataset is empty

    Each grid cell shows:
    - The original image
    - Ground truth OBB annotations with class labels and angles
    - Filename as title (if dataset.file_list exists)
    """
    if len(dataset) == 0:
        print("[visualize_dataset] Dataset is empty")
        return

    # Setup grid layout
    idxs = random.sample(range(len(dataset)), min(num_images, len(dataset)))
    cols = math.ceil(math.sqrt(len(idxs)))
    rows = math.ceil(len(idxs) / cols)

    # Create figure and hide unused subplots
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    axes = np.asarray(axes).reshape(-1)
    for ax in axes[len(idxs) :]:
        ax.axis("off")

    # Plot each sample
    for ax, idx in zip(axes, idxs):
        sample = dataset[idx]
        img_t = sample["image"]
        img_disp = denormalize_image(img_t) if torch.is_tensor(img_t) else img_t
        ax.imshow(img_disp)
        ax.axis("off")
        ax.set_aspect("equal")

        # Draw ground truth boxes
        boxes = sample["target"]["boxes"].cpu().numpy()
        angles = sample["target"]["angles"].cpu().numpy()
        cls_ids = sample["target"]["class_idx"].cpu().numpy()

        for box, ang, cls in zip(boxes, angles, cls_ids):
            draw_obb(ax, box, ang, cls, labels_map)

        # Add filename as title if available
        fname = getattr(dataset, "file_list", [f"img_{idx}"])[idx]
        ax.set_title(Path(fname).name, fontsize=11, color="black")

    plt.tight_layout()
    if show:
        plt.show()
    return fig


def visualize_and_save_dataset_in_script(
    dataset,
    split_name: str,
    save_dir: str,
    num_images: int = 9,
    labels_map: Optional[Dict[int, str]] = None,
):
    """
    Visualizes a sample of the dataset and saves the result as a grid image.

    Args:
        dataset (Dataset): PyTorch dataset with 'image' and 'target' keys.
        split_name (str): Name of the dataset split (e.g., 'train', 'val', 'test').
        save_dir (str): Path to directory where image will be saved.
        num_images (int): Number of images to display.
        labels_map (Optional[Dict[int, str]]): Mapping from class indices to human-readable labels.
    Returns:
        None: Saves the visualization grid to the specified directory.
    """
    fig = visualize_dataset(dataset, num_images=num_images, labels_map=labels_map)
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


def visualize_adultfaces_grid(
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
