import math
import random
import re
import os
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Any

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Polygon
from PIL import Image, ImageDraw, ImageFont, Image
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib import patches
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    confusion_matrix,
    f1_score,
)

from loss.utils import xyxyxyxy2xywhr, xywhr2xyxyxyxy, decode_vertices
from utils.helpers import to_numpy, ensure_polygons_42_shape


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

    pred_xy = decode_vertices(pred_obbs[0], anchors[0], (W, H), clamp_mode="image")
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


def xywhr_to_poly42_shape(
    cx: float, cy: float, w: float, h: float, theta: float
) -> np.ndarray:
    """
    Converts a rotated bounding box defined by center, size, and angle
    to a set of 4 polygon vertices (4, 2).

    The vertices are ordered typically as: top-left, top-right, bottom-right, bottom-left.

    Args:
        cx: Center x-coordinate.
        cy: Center y-coordinate.
        w: Width of the box.
        h: Height of the box.
        theta: Rotation angle (in radians).

    Returns:
        A NumPy array of shape (4, 2) and dtype float32, representing the
        four vertices of the rotated polygon.
    """
    # Calculate half-dimensions for local coordinates
    dx, dy = w / 2.0, h / 2.0

    # Pre-calculate trigonometric values for rotation matrix
    c, s = math.cos(theta), math.sin(theta)

    # Base vertices in the local, unrotated frame (center at 0, 0)
    # Order: Top-Left, Top-Right, Bottom-Right, Bottom-Left
    base = [(-dx, -dy), (dx, -dy), (dx, dy), (-dx, dy)]

    pts = []
    # Apply rotation and translation to each base vertex
    for x, y in base:
        # Rotation: x' = x*cos(theta) - y*sin(theta)
        #           y' = x*sin(theta) + y*cos(theta)
        # Then translation: x_final = cx + x', y_final = cy + y'
        px = cx + x * c - y * s
        py = cy + x * s + y * c
        pts.append((px, py))

    # Return the vertices as a (4, 2) NumPy array with float32 type
    return np.asarray(pts, dtype=np.float32)


def draw_predictions_on_image(
    base_img: np.ndarray,  # (H, W, 3) image in uint8 format
    polygons_xy: np.ndarray,  # (N, 4, 2) coordinates relative to base_img
    labels: np.ndarray,  # (N,) integer labels
    scores: np.ndarray,  # (N,) prediction confidence scores
    angles_rad: np.ndarray,  # (N,) rotation angles in radians (e.g., from AngleHead)
    labels_map: Dict[int, str],  # Map from integer label ID to string name
) -> np.ndarray:
    """
    Renders oriented bounding box (OBB) predictions onto an image using Matplotlib,
    and returns the result as a NumPy array.

    This function sets up a Matplotlib figure exactly matching the input image
    dimensions and uses plotting patches to visualize the OBBs, labels, and scores.
    The final figure is captured from the canvas and returned as a standard
    (H, W, 3) NumPy array.

    Args:
        base_img: The source image data as a NumPy array (H, W, 3, uint8).
        polygons_xy: The vertices of the rotated bounding boxes (N, 4, 2) in image coordinates.
        labels: The class index for each prediction (N,).
        scores: The confidence score for each prediction (N,).
        angles_rad: The rotation angle (in radians) for each box (N,).
        labels_map: A dictionary mapping label IDs to human-readable names.

    Returns:
        The annotated image as a NumPy array (H, W, 3, uint8).
    """
    # 1. Input Check
    # If no polygons are provided, return a copy of the base image immediately.
    if polygons_xy is None or len(polygons_xy) == 0:
        return base_img.copy()

    H, W = int(base_img.shape[0]), int(base_img.shape[1])

    # 2. Matplotlib Figure Setup
    # Create a figure exactly the size of the image in pixels (assuming 100 DPI).
    dpi = 100
    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)
    canvas = FigureCanvas(fig)
    ax = fig.add_axes([0, 0, 1, 1])  # Use the entire canvas area

    # Display the base image
    ax.imshow(base_img, extent=(0, W, H, 0), interpolation="nearest")
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)  # Invert Y axis for standard image coordinates (top-down)
    ax.axis("off")  # Hide axes, ticks, and labels

    # Ensure inputs are standardized NumPy arrays (robustness)
    polys = np.asarray(polygons_xy, dtype=np.float32)
    lbls = np.asarray(labels)
    scrs = np.asarray(scores, dtype=np.float32)
    angs = np.asarray(angles_rad, dtype=np.float32)

    # 3. Draw Predictions
    for i in range(polys.shape[0]):
        coords = polys[i]  # (4,2) vertices for the current box

        # Draw OBB contour (Solid Blue: #004080)
        ax.add_patch(
            Polygon(coords, closed=True, fill=False, edgecolor="#004080", linewidth=1.5)
        )
        # Highlight front edge (v0 -> v1) (Solid Dark Red: #800000)
        ax.plot(coords[[0, 1], 0], coords[[0, 1], 1], color="#800000", linewidth=1.5)

        # 4. Add Text Label
        # Find the top-left corner (TL) of the axis-aligned bounding box (AABB)
        tl_x, tl_y = float(coords[:, 0].min()), float(coords[:, 1].min())

        # Format label text (Name, Angle, Score)
        # Get angle safely, defaulting to 0.0 if array size mismatch occurs
        ang_deg = math.degrees(float(angs[i])) if angs.size > i else 0.0
        name = labels_map.get(int(lbls[i]), str(int(lbls[i])))
        txt = f"{name}: {ang_deg:.1f}° / {float(scrs[i]):.2f}"

        # Draw text with a dark blue background box for improved legibility
        ax.text(
            tl_x,
            tl_y,
            txt,
            color="white",
            fontsize=6,
            ha="left",
            va="top",
            bbox=dict(facecolor="#004080", alpha=0.9, edgecolor="none", pad=2.5),
        )

    # 5. Render and Capture the Figure
    # Force the canvas to render all drawing elements
    canvas.draw()

    # Get the actual dimensions rendered by Matplotlib
    w, h = canvas.get_width_height()

    # Capture the RGB buffer from the canvas and reshape it into a NumPy array (H, W, 3)
    buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    img_annot = buf.reshape(h, w, 3).copy()

    # Crucial step: close the figure to free up memory
    plt.close(fig)

    # 6. Robustness Check: Resize if Matplotlib's output size slightly differs from base_img size
    if (h != H) or (w != W):
        # Resize using PIL's bilinear sampling for better quality
        img_annot = np.asarray(
            Image.fromarray(img_annot).resize((W, H), resample=Image.BILINEAR)
        )

    return img_annot


def write_predictions_txt(
    out_labels_dir: Path,
    stem: str,
    boxes_xywhr: Optional[np.ndarray],  # (N,5) -> cx,cy,w,h,theta(rad)
    polygons_42: Optional[np.ndarray],  # (N,4,2) in the SAME scale as the saved image
    labels: Optional[np.ndarray],  # (N,)
    scores: Optional[np.ndarray],  # (N,)
) -> None:
    """
    Writes object detection predictions to a text file in a standardized format.

    Saves predictions including polygon vertices, class labels, angles and confidence scores.
    If polygon vertices are not provided but boxes are, reconstructs polygons from the boxes.
    All coordinates are saved in the same coordinate system as the input arrays.

    Args:
        out_labels_dir (Path): Directory where the output text files will be saved
        stem (str): Base filename (without extension) for the output .txt file
        boxes_xywhr (Optional[np.ndarray]): Array of shape (N,5) containing box parameters
            [center_x, center_y, width, height, rotation_angle] for N predictions
        polygons_42 (Optional[np.ndarray]): Array of shape (N,4,2) containing vertex
            coordinates for N predictions, each with 4 vertices (x,y)
        labels (Optional[np.ndarray]): Array of shape (N,) containing class indices
        scores (Optional[np.ndarray]): Array of shape (N,) containing confidence scores

    Output format per line:
        <class_id> x1 y1 x2 y2 x3 y3 x4 y4 angle_rad score

    Notes:
        - Creates output directory if it doesn't exist
        - Writes empty file if no predictions exist
        - Coordinates are rounded to integers
        - Angles are saved with 6 decimal precision
        - Default values: labels=0, scores=0.0 if not provided
    """
    # Create output directory if needed
    out_labels_dir.mkdir(parents=True, exist_ok=True)
    txt_path = out_labels_dir / f"{stem}.txt"

    # Convert inputs to numpy arrays with appropriate dtypes
    boxes_np = to_numpy(boxes_xywhr) if boxes_xywhr is not None else None
    labels_np = to_numpy(labels).astype(np.int64) if labels is not None else None
    scores_np = to_numpy(scores).astype(np.float32) if scores is not None else None
    polys_42 = (
        ensure_polygons_42_shape(polygons_42) if polygons_42 is not None else None
    )

    # Reconstruct polygons from boxes if polygons not provided but boxes are
    if (polys_42 is None or polys_42.size == 0) and (
        boxes_np is not None and boxes_np.size > 0
    ):
        N = boxes_np.shape[0]
        polys_42 = np.zeros((N, 4, 2), dtype=np.float32)
        for i in range(N):
            cx, cy, w, h, th = boxes_np[i].tolist()
            polys_42[i] = xywhr_to_poly42_shape(cx, cy, w, h, th)

    # Write empty file if no predictions exist
    if polys_42 is None or polys_42.size == 0:
        with open(txt_path, "w"):
            pass
        return

    # Ensure all arrays have consistent length N
    N = polys_42.shape[0]
    if boxes_np is not None and boxes_np.size > 0:
        boxes_np = boxes_np[:N]
    if labels_np is not None and labels_np.size > 0:
        labels_np = labels_np[:N]
    if scores_np is not None and scores_np.size > 0:
        scores_np = scores_np[:N]

    # Use default values if labels/scores not provided
    if labels_np is None or labels_np.size == 0:
        labels_np = np.zeros((N,), dtype=np.int64)
    if scores_np is None or scores_np.size == 0:
        scores_np = np.zeros((N,), dtype=np.float32)

    # Get angles from boxes if available, else use 0
    if boxes_np is not None and boxes_np.size > 0:
        angles_rad = boxes_np[:, 4]
    else:
        angles_rad = np.zeros((N,), dtype=np.float32)

    # Write predictions to file
    with open(txt_path, "w") as f:
        for i in range(N):
            # Extract vertex coordinates
            x1, y1 = polys_42[i, 0]
            x2, y2 = polys_42[i, 1]
            x3, y3 = polys_42[i, 2]
            x4, y4 = polys_42[i, 3]
            # Write line with class, vertices, angle, score
            f.write(
                f"{int(labels_np[i])} "
                f"{int(round(x1))} {int(round(y1))} "
                f"{int(round(x2))} {int(round(y2))} "
                f"{int(round(x3))} {int(round(y3))} "
                f"{int(round(x4))} {int(round(y4))} "
                f"{float(angles_rad[i]):.6f} {float(scores_np[i]):.6f}\n"
            )


def scale_xywhr_boxes(boxes_np: np.ndarray, sx: float, sy: float) -> np.ndarray:
    """
    Scale oriented bounding boxes from resized coordinates back to original image scale.

    Args:
        boxes_np: Array of shape (N,5) containing [center_x, center_y, width, height, theta]
                 in resized image coordinates, or None/empty array
        sx: Scale factor for x-coordinates (original_width / resized_width)
        sy: Scale factor for y-coordinates (original_height / resized_height)

    Returns:
        Scaled boxes array of same shape as input, with coordinates in original image scale.
        Returns None/empty if input is None/empty.
        Note: Rotation angle (theta) remains unchanged.
    """
    if boxes_np is None or boxes_np.size == 0:
        return boxes_np

    # Create copy to avoid modifying input
    out = boxes_np.copy()

    # Scale center coordinates and dimensions
    out[:, 0] *= sx  # center x
    out[:, 1] *= sy  # center y
    out[:, 2] *= sx  # width
    out[:, 3] *= sy  # height
    # Angle (out[:, 4]) remains unchanged since rotation is scale-invariant

    return out


def scale_polys(
    polys_42: Optional[np.ndarray], sx: float, sy: float
) -> Optional[np.ndarray]:
    """
    Scale polygon vertex coordinates from resized scale back to original image scale.

    Args:
        polys_42: Array of shape (N,4,2) containing N polygons with 4 vertices each,
                 where vertices are [x,y] coordinates in resized image scale,
                 or None/empty array
        sx: Scale factor for x-coordinates (original_width / resized_width)
        sy: Scale factor for y-coordinates (original_height / resized_height)

    Returns:
        Scaled polygon array of same shape as input, with vertices in original image scale.
        Returns None/empty if input is None/empty.
    """
    if polys_42 is None or polys_42.size == 0:
        return polys_42

    # Create copy to avoid modifying input
    out = polys_42.copy()

    # Scale x and y coordinates independently
    out[:, :, 0] *= sx  # All x coordinates
    out[:, :, 1] *= sy  # All y coordinates

    return out


# Function to get image size
def img_size(p: Path) -> Tuple[int, int]:
    with Image.open(p) as im:
        return im.size


def order_polygon_vertices_tl_tr_br_bl(poly42: np.ndarray) -> np.ndarray:
    """
    Orders polygon vertices in clockwise direction starting from top-left.

    Given a polygon with 4 vertices in arbitrary order, this function reorders them to:
    - TL: Top-Left
    - TR: Top-Right
    - BR: Bottom-Right
    - BL: Bottom-Left

    The algorithm uses sum and difference of coordinates to robustly identify corners:
    - TL has minimum sum of coordinates (x+y)
    - BR has maximum sum of coordinates (x+y)
    - TR has maximum difference of coordinates (x-y)
    - BL has minimum difference of coordinates (x-y)

    Args:
        poly42: Input polygon vertices as (4,2) numpy array

    Returns:
        Reordered vertices as (4,2) float32 array in [TL,TR,BR,BL] order.
        If ordering fails, returns input array unchanged.
    """
    # Convert to float32 for numerical stability
    p = poly42.astype(np.float32)

    # Calculate sum (x+y) and difference (x-y) for each vertex
    coord_sums = p.sum(1)  # For finding TL (min) and BR (max)
    coord_diffs = p[:, 0] - p[:, 1]  # For disambiguating TR/BL

    # Find top-left and bottom-right vertices
    tl = p[np.argmin(coord_sums)]
    br = p[np.argmax(coord_sums)]

    # Get indices of remaining two vertices (not TL or BR)
    remaining_idx = [
        i for i in range(4) if not np.allclose(p[i], tl) and not np.allclose(p[i], br)
    ]

    # Order remaining vertices as TR/BL based on x-y difference
    if len(remaining_idx) == 2:
        r0, r1 = remaining_idx
        # Larger difference (x-y) is TR, smaller is BL
        tr = p[r0] if coord_diffs[r0] > coord_diffs[r1] else p[r1]
        bl = p[r1] if coord_diffs[r0] > coord_diffs[r1] else p[r0]
        return np.stack([tl, tr, br, bl], axis=0).astype(np.float32)

    # Fallback: return original array if ordering fails
    return p


def get_oriented_face_crop(
    base_img: np.ndarray,
    poly42: np.ndarray,  # (4,2) vertex coordinates in base_img space
    angle_rad: float,  # predicted rotation angle (radians, CCW+)
    pivot: str = "tl",  # rotation pivot point
    crop_out_wh: tuple = (640, 640),  # output (width,height)
    border_mode: str = "replicate",  # border handling mode
    scale_crop: float = 1.0,  # >1.0 adds margin around OBB axes
) -> Optional[np.ndarray]:
    """
    Extracts an oriented face crop from an image by aligning it to be upright.

    This function performs three key steps:
    1. Rotates the entire image by angle_rad (deskew) with canvas expansion
    2. Transforms the oriented bounding box (OBB) with the same affine matrix
    3. Uses perspective warping to map the OBB to a rectangular output
       (Avoids axis-aligned bounding box issues and fragile W/H swaps)

    Args:
        base_img: Source image as numpy array (H,W,C)
        poly42: OBB vertices as (4,2) array in source image coordinates
        angle_rad: Rotation angle in radians (positive = counterclockwise)
        pivot: Rotation pivot point ("center" or "top-left")
        crop_out_wh: Output dimensions as (width,height) tuple
        border_mode: Border handling ("replicate", "black", or "white")
        scale_crop: Scale factor for output crop (>1.0 adds margin)

    Returns:
        Warped and cropped image array, or None if operation fails
    """
    assert poly42.shape == (4, 2)
    H, W = base_img.shape[:2]

    # --- 1) Global rotation ---
    angle_deg_ccw = float(math.degrees(float(angle_rad)))

    # Choose stable pivot point
    if pivot == "center":
        px, py = float(poly42[:, 0].mean()), float(poly42[:, 1].mean())
    else:
        ordered0 = order_polygon_vertices_tl_tr_br_bl(poly42)
        px, py = float(ordered0[0, 0]), float(ordered0[0, 1])

    # Get rotation matrix around pivot
    # OpencCV uses CCW+ angle convention
    R = cv2.getRotationMatrix2D((px, py), angle_deg_ccw, 1.0)

    # Calculate expanded canvas size
    corners = np.array([[0, 0, 1], [W, 0, 1], [W, H, 1], [0, H, 1]], dtype=np.float32).T
    rot_xy = (R @ corners).T[:, :2]
    min_xy, max_xy = rot_xy.min(0), rot_xy.max(0)
    tx, ty = -min_xy[0], -min_xy[1]
    R[:, 2] += [tx, ty]  # Add translation to matrix
    newW = int(math.ceil(max_xy[0] - min_xy[0]))
    newH = int(math.ceil(max_xy[1] - min_xy[1]))

    # Set border handling mode
    if border_mode == "replicate":
        bmode, bval = cv2.BORDER_REPLICATE, 0
    elif border_mode == "white":
        bmode, bval = cv2.BORDER_CONSTANT, (255, 255, 255)
    else:
        bmode, bval = cv2.BORDER_CONSTANT, (0, 0, 0)

    # Apply rotation to full image
    rot_img = cv2.warpAffine(
        base_img,
        R,
        (newW, newH),
        flags=cv2.INTER_LINEAR,
        borderMode=bmode,
        borderValue=bval,
    )

    # --- 2) Transform polygon with same affine matrix ---
    poly_h = np.concatenate(
        [poly42.astype(np.float32), np.ones((4, 1), np.float32)], axis=1
    )
    rot_poly = (R @ poly_h.T).T[:, :2]  # Apply rotation

    # Get canonical vertex ordering (TL,TR,BR,BL) after rotation
    p = order_polygon_vertices_tl_tr_br_bl(rot_poly).astype(np.float32)

    # --- 3) Exact perspective warp of OBB ---
    # Calculate sides in polygon space (avoids aspect ratio issues)
    w_side = float(np.linalg.norm(p[1] - p[0]))  # |TR - TL|
    h_side = float(np.linalg.norm(p[3] - p[0]))  # |BL - TL|

    # Apply margin scale factor
    s = max(1.0, float(scale_crop))
    W_src = max(1.0, w_side * s)
    H_src = max(1.0, h_side * s)

    # Expand OBB around center while maintaining directions
    c = p.mean(0)  # center point
    u = p[1] - p[0]  # width axis
    v = p[3] - p[0]  # height axis
    u_n = u / (np.linalg.norm(u) + 1e-9)  # normalized width vector
    v_n = v / (np.linalg.norm(v) + 1e-9)  # normalized height vector
    halfW, halfH = 0.5 * W_src, 0.5 * H_src

    # Calculate expanded corners
    tl = c - halfW * u_n - halfH * v_n
    tr = c + halfW * u_n - halfH * v_n
    br = c + halfW * u_n + halfH * v_n
    bl = c - halfW * u_n + halfH * v_n
    src = np.stack([tl, tr, br, bl], axis=0).astype(np.float32)

    # Define target rectangle coords
    Wt, Ht = crop_out_wh
    dst = np.array(
        [[0, 0], [Wt - 1, 0], [Wt - 1, Ht - 1], [0, Ht - 1]], dtype=np.float32
    )

    # Apply perspective transform
    M = cv2.getPerspectiveTransform(src, dst)
    out = cv2.warpPerspective(
        rot_img, M, (Wt, Ht), flags=cv2.INTER_LINEAR, borderMode=bmode, borderValue=bval
    )

    return out


def plot_gt_angle_histograms_counts(
    gt_angles_all_deg: List[float],
    gt_angles_per_cls_deg: Dict[int, List[float]],
    labels_map: Dict[int, str],
    bin_deg: int = 10,
) -> Dict[str, plt.Figure]:
    """
    Create histogram figures of ground-truth face angles (degrees).

    This helper builds two matplotlib figures:
      - "all": a single histogram aggregating all GT angles across classes.
      - "per_class": a grid of histograms, one per class (ordered by labels_map keys).

    Purpose:
      - Inspect the angular distribution of annotated faces.
      - Reveal class imbalances or preferred orientations in the dataset.

    Arguments:
      gt_angles_all_deg: Flat list of GT angles in degrees in range [0, 180).
      gt_angles_per_cls_deg: Mapping class_index -> list of GT angles (degrees).
      labels_map: Mapping from class_index -> human readable class name.
      bin_deg: Histogram bin width in degrees. Must be in (0, 180].

    Returns:
      Dict with keys:
        - "all": Figure with aggregated histogram.
        - "per_class": Figure with per-class histogram grid.

    Notes:
      - Bins are generated as np.arange(0, 180 + bin_deg, bin_deg) so the last bin
        includes angles close to 180 degrees. Angles should already be in degrees.
      - Empty classes produce empty histograms (count = 0) and are still shown
        in the grid; axes for unused grid cells are turned off.
    """
    # Validate bin width
    assert 0 < bin_deg <= 180, "bin_deg must be in the interval (0, 180]"

    # Prepare bin edges from 0 to 180 inclusive so bins represent [0, bin_deg), ... ,[180-bin_deg,180]
    bins = np.arange(0, 180 + bin_deg, bin_deg)

    # -------------------------
    # Aggregated histogram (all classes combined)
    # -------------------------
    fig_all, ax_all = plt.subplots(figsize=(8, 4.5))
    # Draw histogram with black edges for better readability
    ax_all.hist(gt_angles_all_deg, bins=bins, edgecolor="black")
    ax_all.set_title(f"GT angle histogram (all samples) — bin={bin_deg}°")
    ax_all.set_xlabel("GT angle [deg] ∈ [0, 180)")
    ax_all.set_ylabel("Count")
    ax_all.grid(axis="y", linestyle=":", alpha=0.6)
    # Remove top/right spines for a cleaner look
    for s in ("top", "right"):
        ax_all.spines[s].set_visible(False)
    fig_all.tight_layout()

    # -------------------------
    # Per-class histograms grid
    # -------------------------
    classes = list(labels_map.keys())
    n_cls = len(classes)
    # Choose up to 3 columns to keep subplots readable; adjust rows accordingly
    n_cols = min(3, n_cls) if n_cls > 0 else 1
    n_rows = math.ceil(n_cls / n_cols) if n_cls > 0 else 1
    fig_cls, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.8 * n_rows))
    # Ensure axes is 2D array for consistent indexing
    axes = np.atleast_2d(axes)

    for idx, c in enumerate(classes):
        r, col = divmod(idx, n_cols)
        ax = axes[r, col]
        vals = gt_angles_per_cls_deg.get(c, [])
        # Plot histogram even if vals is empty (will render empty axes)
        ax.hist(vals, bins=bins, edgecolor="black")
        ax.set_title(f"{labels_map[c]} (n={len(vals)}) — bin={bin_deg}°")
        ax.set_xlabel("GT angle [deg]")
        ax.set_ylabel("Count")
        ax.grid(axis="y", linestyle=":", alpha=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    # Turn off any unused axes (when grid larger than number of classes)
    total_cells = n_rows * n_cols
    for k in range(n_cls, total_cells):
        r, col = divmod(k, n_cols)
        axes[r, col].axis("off")

    fig_cls.suptitle(f"GT angle histogram per class — bin={bin_deg}°")
    # Leave space for the suptitle
    fig_cls.tight_layout(rect=[0, 0, 1, 0.97])

    return {"all": fig_all, "per_class": fig_cls}


def full_bin_ids(bin_deg: int, max_deg: int = 180) -> List[int]:
    """
    Return the sequence of integer bin indices that partition the interval [0, max_deg)
    into contiguous bins of width `bin_deg`.

    The number of bins K is computed as ceil(max_deg / bin_deg) so that the union of
    bins [0, bin_deg), [bin_deg, 2*bin_deg), ..., [(K-1)*bin_deg, K*bin_deg) covers
    the range up to (and possibly beyond) max_deg. Only indices 0..K-1 are returned.

    Args:
        bin_deg: Width of each bin in degrees. Must be positive.
        max_deg: Upper bound (exclusive) of the angle range to cover. Default 180.

    Returns:
        List[int]: Consecutive bin indices [0, 1, ..., K-1].

    Raises:
        ValueError: If bin_deg <= 0 or max_deg <= 0.

    Examples:
        >>> full_bin_ids(10, 180)
        [0, 1, 2, ..., 17]
        >>> full_bin_ids(45, 100)
        [0, 1, 2]
    """
    if bin_deg <= 0:
        raise ValueError("bin_deg must be > 0")
    if max_deg <= 0:
        raise ValueError("max_deg must be > 0")

    # Compute number of bins required to cover [0, max_deg).
    k = int(math.ceil(float(max_deg) / float(bin_deg)))

    # Return integer indices for each bin: 0 .. k-1
    return list(range(k))


def plot_error_box_by_gt_bins(
    errs_by_bin: Dict[int, List[float]],
    bin_deg: int,
    title: str,
    y_lim: Tuple[float, float] = (0, 180),
    show_counts: bool = True,
    show_global_mean: bool = True,
):
    """
    Creates boxplots of angular errors grouped by ground truth (GT) angle bins.

    This function generates two types of boxplots:
    1. `fig_noempty`: Includes only bins with data.
    2. `fig_all`: Includes all bins [0, 180) based on `bin_deg`, showing empty bins
       as gaps (no box) but with labels and `n=0` annotations.

    Args:
        errs_by_bin (Dict[int, List[float]]): Mapping of bin indices to lists of angular errors.
        bin_deg (int): Width of each bin in degrees (e.g., 10 for bins like [0,10), [10,20), etc.).
        title (str): Title for the plots.
        y_lim (Tuple[float, float], optional): Y-axis limits for the plots. Default is (0, 180).
        show_counts (bool, optional): Whether to annotate each box with the sample count. Default is True.
        show_global_mean (bool, optional): Whether to display a horizontal line for the global mean. Default is True.

    Returns:
        Tuple[plt.Figure, plt.Figure]: A tuple containing:
            - `fig_noempty`: Boxplot figure with only bins that have data.
            - `fig_all`: Boxplot figure with all bins, including empty ones.

    Notes:
        - The `fig_noempty` plot excludes bins with no data.
        - The `fig_all` plot includes all bins, even if they are empty, and annotates them with `n=0`.
        - The global mean is calculated across all bins and displayed as a dashed line if `show_global_mean` is True.
    """
    # ---------- Variant A: Only bins with data ----------
    # Filter bins to include only those with data
    bin_ids = [b for b in sorted(errs_by_bin.keys()) if len(errs_by_bin[b]) > 0]
    values = [errs_by_bin[b] for b in bin_ids]
    labels = [f"[{b*bin_deg},{(b+1)*bin_deg})" for b in bin_ids]

    # Calculate global mean across all values
    all_vals = [v for lst in values for v in lst]
    global_mean = float(np.mean(all_vals)) if all_vals else 0.0

    # Create the figure for bins with data
    fig_w = max(9, 1.1 * max(1, len(labels)))  # Adjust width based on number of bins
    fig_noempty, ax = plt.subplots(figsize=(fig_w, 5.2))

    if values:  # Only plot if there are bins with data
        pos = np.arange(len(values))  # Positions for the boxplots
        bp = ax.boxplot(
            values,
            positions=pos,
            notch=True,
            patch_artist=True,
            widths=0.7,
            boxprops=dict(facecolor="none", edgecolor="0.2"),
            medianprops=dict(color="white", linewidth=1.8),
            whiskerprops=dict(color="0.4"),
            capprops=dict(color="0.4"),
            flierprops=dict(
                marker="o",
                markersize=3,
                markerfacecolor="0.6",
                markeredgecolor="0.4",
                alpha=0.7,
            ),
        )

        # Apply colors to the boxes
        cmap = plt.get_cmap("tab10")
        colors = [cmap(i % cmap.N) for i in range(len(values))]
        for i, box in enumerate(bp["boxes"]):
            box.set(facecolor=colors[i], edgecolor="0.15", linewidth=0.9, alpha=0.95)
        for med in bp["medians"]:
            med.set(color="white", linewidth=1.6)

        # Set x-axis labels and limits
        ax.set_xticks(pos)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
        ax.set_xlim(-0.5, len(pos) - 0.5)  # No extra margin on the sides

        # Annotate sample counts above each box
        if show_counts:
            pad = 0.02 * ((y_lim[1] - y_lim[0]) if y_lim else 100.0)  # Vertical padding
            for i, vals in enumerate(values):
                n = len(vals)
                y = (float(np.nanmax(vals)) if n > 0 else 0.0) + pad
                ax.text(i, y, f"n={n}", ha="center", va="bottom", fontsize=9)
    else:
        # If no data, display a "no data" message
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])

    # Set axis labels, title, and grid
    ax.set_ylabel("Angular error [deg]", fontsize=11)
    ax.set_title(f"{title}  (bin={bin_deg}°)", fontsize=12)
    if y_lim is not None:
        ax.set_ylim(*y_lim)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.tick_params(axis="y", labelsize=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # Add global mean line if requested
    if show_global_mean and all_vals:
        ax.axhline(global_mean, linestyle="--", linewidth=1.2, color="gray", alpha=0.9)
        ax.text(
            0.01,
            0.97,
            f"Global mean = {global_mean:.1f}°",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            color="gray",
        )

    fig_noempty.tight_layout()

    # ---------- Variant B: All bins (including empty ones) ----------
    # Generate all bins, including empty ones
    full_bins = full_bin_ids(bin_deg, 180)
    labels_all = [f"[{b*bin_deg},{(b+1)*bin_deg})" for b in full_bins]

    # Prepare data for all bins
    values_all = [errs_by_bin.get(b, []) for b in full_bins]
    pos_all = [i for i, v in enumerate(values_all) if len(v) > 0]  # Positions with data
    draw_vals = [v for v in values_all if len(v) > 0]  # Values for bins with data

    # Calculate global mean for all bins
    all_vals2 = [v for lst in values_all for v in lst]
    global_mean2 = float(np.mean(all_vals2)) if all_vals2 else 0.0

    # Create the figure for all bins
    fig_all, ax2 = plt.subplots(figsize=(max(9, 1.1 * len(labels_all)), 5.2))
    if draw_vals:  # Only plot if there are bins with data
        bp2 = ax2.boxplot(
            draw_vals,
            positions=np.array(pos_all, dtype=float),
            notch=True,
            patch_artist=True,
            widths=0.7,
            boxprops=dict(facecolor="none", edgecolor="0.2"),
            medianprops=dict(color="white", linewidth=1.8),
            whiskerprops=dict(color="0.4"),
            capprops=dict(color="0.4"),
            flierprops=dict(
                marker="o",
                markersize=3,
                markerfacecolor="0.6",
                markeredgecolor="0.4",
                alpha=0.7,
            ),
        )

        # Apply colors to the boxes
        cmap = plt.get_cmap("tab10")
        colors2 = [cmap(i % cmap.N) for i in range(len(draw_vals))]
        for i, box in enumerate(bp2["boxes"]):
            box.set(facecolor=colors2[i], edgecolor="0.15", linewidth=0.9, alpha=0.95)
        for med in bp2["medians"]:
            med.set(color="white", linewidth=1.6)

    # Set x-axis labels and limits
    ax2.set_xticks(np.arange(len(full_bins)))
    ax2.set_xticklabels(labels_all, rotation=45, ha="right", fontsize=10)
    ax2.set_xlim(-0.5, len(full_bins) - 0.5)  # No extra margin on the sides

    # Annotate sample counts above each box
    if show_counts:
        pad2 = 0.02 * ((y_lim[1] - y_lim[0]) if y_lim else 100.0)  # Vertical padding
        for i, vals in enumerate(values_all):
            n = len(vals)
            y = (float(np.nanmax(vals)) if n > 0 else 0.0) + pad2
            ax2.text(i, y, f"n={n}", ha="center", va="bottom", fontsize=9, color="0.3")

    # Set axis labels, title, and grid
    ax2.set_ylabel("Angular error [deg]", fontsize=11)
    ax2.set_title(f"{title}  (bin={bin_deg}°) — all bins", fontsize=12)
    if y_lim is not None:
        ax2.set_ylim(*y_lim)
    ax2.grid(axis="y", linestyle=":", alpha=0.6)
    ax2.tick_params(axis="y", labelsize=10)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)

    # Add global mean line if requested
    if show_global_mean and all_vals2:
        ax2.axhline(
            global_mean2, linestyle="--", linewidth=1.2, color="gray", alpha=0.9
        )
        ax2.text(
            0.01,
            0.97,
            f"Global mean = {global_mean2:.1f}°",
            transform=ax2.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            color="gray",
        )

    fig_all.tight_layout()

    return fig_noempty, fig_all


def plot_error_bar_mean_std_by_gt_bins(
    errs_by_bin: Dict[int, List[float]],
    bin_deg: int,
    title: str,
    y_lim: Tuple[float, float] = (0, 180),
    show_counts: bool = True,
    show_global_mean: bool = True,
):
    """
    Creates bar plots (mean ± std) of angular errors grouped by ground truth (GT) angle bins.

    This function generates two types of bar plots:
    1. `fig_noempty`: Includes only bins with data.
    2. `fig_all`: Includes all bins [0, 180) based on `bin_deg`, showing empty bins
       as gaps (no bar) but with labels and `n=0` annotations.

    Args:
        errs_by_bin (Dict[int, List[float]]): Mapping of bin indices to lists of angular errors.
        bin_deg (int): Width of each bin in degrees (e.g., 10 for bins like [0,10), [10,20), etc.).
        title (str): Title for the plots.
        y_lim (Tuple[float, float], optional): Y-axis limits for the plots. Default is (0, 180).
        show_counts (bool, optional): Whether to annotate each bar with the sample count. Default is True.
        show_global_mean (bool, optional): Whether to display a horizontal line for the global mean. Default is True.

    Returns:
        Tuple[plt.Figure, plt.Figure]: A tuple containing:
            - `fig_noempty`: Bar plot figure with only bins that have data.
            - `fig_all`: Bar plot figure with all bins, including empty ones.

    Notes:
        - The `fig_noempty` plot excludes bins with no data.
        - The `fig_all` plot includes all bins, even if they are empty, and annotates them with `n=0`.
        - The global mean is calculated across all bins and displayed as a dashed line if `show_global_mean` is True.
    """

    def stats(vec: List[float]) -> Tuple[float, float, int]:
        """
        Computes the mean, standard deviation, and count of a list of values.

        Args:
            vec (List[float]): List of values.

        Returns:
            Tuple[float, float, int]: Mean, standard deviation, and count of the values.
        """
        return (
            float(np.mean(vec)) if vec else 0.0,
            float(np.std(vec, ddof=0)) if vec else 0.0,
            len(vec),
        )

    # ---------- Variant A: Only bins with data ----------
    # Filter bins to include only those with data
    bin_ids = [b for b in sorted(errs_by_bin.keys()) if len(errs_by_bin[b]) > 0]
    means, stds, ns = [], [], []
    for b in bin_ids:
        m, s, n = stats(errs_by_bin[b])
        means.append(m)
        stds.append(s)
        ns.append(n)

    labels = [f"[{b*bin_deg},{(b+1)*bin_deg})" for b in bin_ids]
    all_vals = [v for b in bin_ids for v in errs_by_bin[b]]
    global_mean = float(np.mean(all_vals)) if all_vals else 0.0

    x = np.arange(len(bin_ids))
    fig_noempty, ax = plt.subplots(figsize=(max(9, 1.1 * max(1, len(labels))), 5.2))
    if len(x) > 0:
        # Create bar plot with error bars
        cmap = plt.get_cmap("tab10")
        colors = [cmap(i % cmap.N) for i in range(len(x))]
        bars = ax.bar(
            x,
            means,
            yerr=stds,
            capsize=5,
            color=colors,
            edgecolor="0.15",
            linewidth=0.8,
            alpha=0.95,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
        ax.set_xlim(-0.5, len(x) - 0.5)  # No extra margin on the sides

        # Annotate sample counts above each bar
        if show_counts:
            pad = 0.02 * ((y_lim[1] - y_lim[0]) if y_lim else 100.0)  # Vertical padding
            for rect, n in zip(bars, ns):
                height = rect.get_height()
                ax.text(
                    rect.get_x() + rect.get_width() / 2,
                    max(height, 0) + pad,
                    f"n={n}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
    else:
        # If no data, display a "no data" message
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])

    # Set axis labels, title, and grid
    ax.set_ylabel("Angular error [deg]", fontsize=11)
    ax.set_title(f"{title}  (bin={bin_deg}°)", fontsize=12)
    if y_lim is not None:
        ax.set_ylim(*y_lim)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.tick_params(axis="y", labelsize=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # Add global mean line if requested
    if show_global_mean and all_vals:
        ax.axhline(global_mean, linestyle="--", linewidth=1.4, color="gray", alpha=0.9)
        ax.text(
            0.01,
            0.97,
            f"Global mean = {global_mean:.1f}°",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            color="gray",
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=2),
        )
    fig_noempty.tight_layout()

    # ---------- Variant B: All bins (including empty ones) ----------
    # Generate all bins, including empty ones
    full_bins = full_bin_ids(bin_deg, 180)
    labels_all = [f"[{b*bin_deg},{(b+1)*bin_deg})" for b in full_bins]
    m_all, s_all, n_all = [], [], []
    for b in full_bins:
        m, s, n = stats(errs_by_bin.get(b, []))
        m_all.append(m)
        s_all.append(s)
        n_all.append(n)

    x2 = np.arange(len(full_bins))
    all_vals2 = [v for b in full_bins for v in errs_by_bin.get(b, [])]
    global_mean2 = float(np.mean(all_vals2)) if all_vals2 else 0.0

    fig_all, ax2 = plt.subplots(figsize=(max(9, 1.1 * len(labels_all)), 5.2))
    cmap2 = plt.get_cmap("tab20")
    colors2 = [cmap2(i % cmap2.N) for i in range(len(x2))]
    bars2 = ax2.bar(
        x2,
        m_all,
        yerr=s_all,
        capsize=5,
        color=colors2,
        edgecolor="0.15",
        linewidth=0.8,
        alpha=0.95,
    )

    ax2.set_xticks(x2)
    ax2.set_xticklabels(labels_all, rotation=45, ha="right", fontsize=10)
    ax2.set_xlim(-0.5, len(x2) - 0.5)  # No extra margin on the sides
    ax2.set_ylabel("Angular error [deg]", fontsize=11)
    ax2.set_title(f"{title}  (bin={bin_deg}°) — all bins", fontsize=12)
    if y_lim is not None:
        ax2.set_ylim(*y_lim)
    ax2.grid(axis="y", linestyle=":", alpha=0.6)
    ax2.tick_params(axis="y", labelsize=10)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)

    # Annotate sample counts above each bar
    if show_counts:
        pad2 = 0.02 * ((y_lim[1] - y_lim[0]) if y_lim else 100.0)  # Vertical padding
        for rect, n in zip(bars2, n_all):
            height = rect.get_height()
            ax2.text(
                rect.get_x() + rect.get_width() / 2,
                max(height, 0) + pad2,
                f"n={n}",
                ha="center",
                va="bottom",
                fontsize=9,
                color="0.3",
            )

    # Add global mean line if requested
    if show_global_mean and all_vals2:
        ax2.axhline(
            global_mean2, linestyle="--", linewidth=1.4, color="gray", alpha=0.9
        )
        ax2.text(
            0.01,
            0.97,
            f"Global mean = {global_mean2:.1f}°",
            transform=ax2.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            color="gray",
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=2),
        )

    fig_all.tight_layout()
    return fig_noempty, fig_all


def plot_error_bar_mean_std_by_gt_bins_per_class(
    errs_by_bin_per_cls: Dict[int, Dict[int, List[float]]],
    labels_map: Dict[int, str],
    bin_deg: int,
    title_prefix: str = "Angular error vs GT angle bin",
    y_lim: Tuple[float, float] = (0, 180),
    share_y_axis: bool = True,
    show_global_mean_each: bool = True,
    show_counts: bool = False,
    cmap_name: str = "tab10",
) -> plt.Figure:
    """
    Create a multi-panel bar plot (mean ± std) of angular errors grouped by GT-angle bins,
    produced separately per class.

    This function expects a nested mapping errs_by_bin_per_cls[class_idx][bin_idx] -> list of
    angular errors (degrees). For each class a subplot is drawn with a bar per bin where:
      - bar height = mean error for the bin
      - error bar   = std deviation for the bin
      - optional: a dashed horizontal line showing the class global mean
      - optional: annotation above each bar with sample count n=...

    Improvements over a minimal implementation:
      - nicer qualitative colormap (configurable with cmap_name)
      - defensive handling of empty classes or empty bins
      - consistent axis styling and optional shared Y limits for easy comparison
      - informative per-class subplot titles using labels_map

    Args:
        errs_by_bin_per_cls: mapping class_idx -> (mapping bin_idx -> list of error values)
        labels_map: mapping class_idx -> human readable class name
        bin_deg: width (degrees) used to label bins (e.g. 10)
        title_prefix: overall figure title prefix
        y_lim: tuple (ymin, ymax) applied to each subplot if provided
        share_y_axis: if True force all used subplots to share the same Y limits
        show_global_mean_each: draw dashed line for class global mean when True
        show_counts: annotate each bar with number of samples (n=...)
        cmap_name: matplotlib colormap name used to color bars

    Returns:
        matplotlib.figure.Figure with one subplot per class (arranged in a grid).
    """
    classes = list(labels_map.keys())
    n_cls = len(classes)
    if n_cls == 0:
        raise ValueError("labels_map must contain at least one class")

    # Layout: up to 3 columns to keep individual plots readable
    n_cols = min(3, n_cls)
    n_rows = math.ceil(n_cls / n_cols)

    # Figure sizing: scale with grid size
    fig_w = 4.6 * n_cols
    fig_h = 3.4 * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False)

    # Colormap for bars; use qualitative colormap and cycle per bin
    cmap = plt.get_cmap(cmap_name)
    # Track which axes actually contain plotted data so we can set shared y-limits
    used_axes = []

    for idx, cls in enumerate(classes):
        r, c = divmod(idx, n_cols)
        ax = axes[r, c]

        errs_by_bin = errs_by_bin_per_cls.get(cls, {})
        bin_ids = sorted(errs_by_bin.keys())

        # If this class has no bins, render a clear "no data" placeholder
        if not bin_ids:
            ax.text(
                0.5, 0.5, "no data", ha="center", va="center", fontsize=10, color="0.4"
            )
            ax.axis("off")
            continue

        # Compute summary statistics per bin, handling empty lists gracefully
        means = [
            float(np.mean(errs_by_bin[b])) if errs_by_bin[b] else 0.0 for b in bin_ids
        ]
        stds = [
            float(np.std(errs_by_bin[b], ddof=0)) if errs_by_bin[b] else 0.0
            for b in bin_ids
        ]
        ns = [len(errs_by_bin[b]) for b in bin_ids]
        labels = [f"[{b*bin_deg},{(b+1)*bin_deg})" for b in bin_ids]
        x = np.arange(len(bin_ids))

        # Choose a set of colors for the bars; rotate through the cmap so adjacent classes differ
        colors = [cmap(i % cmap.N) for i in range(len(bin_ids))]

        # Draw bars with errorbars and nicer styling
        bars = ax.bar(
            x,
            means,
            yerr=stds,
            capsize=4,
            color=colors,
            edgecolor="0.15",
            linewidth=0.8,
            alpha=0.92,
        )

        # Annotate sample counts above each bar (optional)
        if show_counts:
            # compute a y-offset that is a small fraction of y-range (fallback if y_lim None)
            yspan = (
                (y_lim[1] - y_lim[0])
                if y_lim is not None
                else (max(means + stds) + 1.0)
            )
            offset = 0.03 * (yspan if yspan > 0 else 10.0)
            for rect, n in zip(bars, ns):
                if n > 0:
                    height = rect.get_height()
                    ax.text(
                        rect.get_x() + rect.get_width() / 2,
                        height + offset,
                        f"n={n}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        color="0.15",
                    )

        # Per-class global mean line (optional)
        all_vals = [v for b in bin_ids for v in errs_by_bin[b]]
        if show_global_mean_each and len(all_vals) > 0:
            gmean = float(np.mean(all_vals))
            ax.axhline(gmean, linestyle="--", linewidth=1.0, color="gray", alpha=0.9)
            ax.text(
                0.98,
                0.92,
                f"mean={gmean:.2f}°",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                color="gray",
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=2),
            )

        # Axis labels and styling
        ax.set_title(f"{labels_map.get(cls, str(cls))}", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("Angular error [deg]", fontsize=10)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

        if y_lim is not None:
            ax.set_ylim(*y_lim)

        used_axes.append(ax)

    # Turn off any remaining empty subplots in the grid
    total_cells = n_rows * n_cols
    for k in range(n_cls, total_cells):
        r, c = divmod(k, n_cols)
        axes[r, c].axis("off")

    # Enforce shared y-limits across used axes if requested
    if share_y_axis and y_lim is not None and used_axes:
        for a in used_axes:
            a.set_ylim(*y_lim)

    # Overall title and layout tweaks
    fig.suptitle(f"{title_prefix}  (bin={bin_deg}°)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def plot_training_curves_from_csv(csv_path: str, output_dir: Path) -> None:
    """
    Creates and saves training/validation curves from a model training log CSV.

    Generates separate plots for different loss components and metrics:
    - Face detection loss
    - Classification loss
    - Angular regression loss
    - Oriented bounding box (OBB) loss
    - Total combined loss
    - Mean Average Precision (mAP)

    Each plot shows training and validation curves (except mAP which is validation only)
    with epochs on x-axis and the corresponding metric on y-axis.

    Args:
        csv_path (str): Path to CSV file containing training logs with columns:
            epoch, train_face_loss, test_face_loss, train_class_loss, etc.
        output_dir (Path): Directory where plots will be saved under 'curves' subfolder.
            Will be created if it doesn't exist.
    """
    # Read training log data
    df = pd.read_csv(csv_path)
    curves_path = output_dir / "curves"
    curves_path.mkdir(parents=True, exist_ok=True)

    def make_plot(train_col: str, val_col: str, title: str, ylabel: str, filename: str):
        """Helper function to create and save a single training/validation curve plot"""
        plt.figure(figsize=(10, 6), facecolor="white")

        # Plot training curve
        plt.plot(
            df["epoch"],
            df[train_col],
            label="Training",
            color="#2ecc71",  # Bright green
            linewidth=2,
            marker="o",
            markersize=6,
            alpha=0.8,
        )

        # Plot validation curve
        plt.plot(
            df["epoch"],
            df[val_col],
            label="Validation",
            color="#e74c3c",  # Bright red
            linewidth=2,
            marker="s",
            markersize=6,
            alpha=0.8,
        )

        # Styling
        plt.xlabel("Epoch", fontsize=12, labelpad=10)
        plt.ylabel(ylabel, fontsize=12, labelpad=10)
        plt.title(title, fontsize=14, pad=15)

        # Grid and background
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.gca().set_facecolor("#f8f9fa")  # Light gray background

        # Legend with semi-transparent background
        plt.legend(
            framealpha=0.95,
            facecolor="white",
            edgecolor="none",
            fontsize=10,
            loc="upper right",
        )

        plt.tight_layout()
        plt.savefig(curves_path / f"{filename}.png", dpi=300, bbox_inches="tight")
        plt.close()

    # Generate individual loss component plots
    make_plot(
        "train_face_loss",
        "test_face_loss",
        "Face Detection Loss Over Training",
        "Loss",
        "face_curves",
    )
    make_plot(
        "train_class_loss",
        "test_class_loss",
        "Classification Loss Over Training",
        "Loss",
        "class_curves",
    )
    make_plot(
        "train_angular_loss",
        "test_angular_loss",
        "Angular Regression Loss Over Training",
        "Loss",
        "angle_curves",
    )
    make_plot(
        "train_obb_loss",
        "test_obb_loss",
        "OBB Regression Loss Over Training",
        "Loss",
        "obb_curves",
    )
    make_plot(
        "train_rect_loss",
        "test_rect_loss",
        "Orthogonality Regularization Over Training",
        "Loss",
        "regularization_curves",
    )
    make_plot(
        "train_child_loss",
        "test_child_loss",
        "Child Loss Over Training",
        "Loss",
        "child_curves",
    )
    make_plot(
        "train_total_loss",
        "test_total_loss",
        "Total Combined Loss Over Training",
        "Loss",
        "total_curves",
    )

    # mAP plot (validation only)
    plt.figure(figsize=(10, 6), facecolor="white")
    plt.plot(
        df["epoch"],
        df["test_mAP"],
        label="Validation mAP",
        color="#3498db",  # Bright blue
        linewidth=2.5,
        marker="D",
        markersize=7,
    )

    plt.xlabel("Epoch", fontsize=12, labelpad=10)
    plt.ylabel("mAP", fontsize=12, labelpad=10)
    plt.title("Mean Average Precision Over Training", fontsize=14, pad=15)

    plt.grid(True, linestyle="--", alpha=0.3)
    plt.gca().set_facecolor("#f8f9fa")
    plt.legend(framealpha=0.95, facecolor="white", edgecolor="none", fontsize=10)

    plt.tight_layout()
    plt.savefig(curves_path / "map.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[INFO] Training curves saved to: {curves_path}")


def smooth_curve(x: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """
    Applies a Gaussian smoothing filter to a 1D array.

    Args:
        x (np.ndarray): Input array to smooth.
        sigma (float): Standard deviation of the Gaussian filter.

    Returns:
        np.ndarray: Smoothed array.
    """
    return gaussian_filter1d(x, sigma=sigma)


def plot_precision_recall(
    per_true: Dict[int, List[int]],
    per_score: Dict[int, List[float]],
    labels_map: Dict[int, str],
    mAP: float,
    sigma: float = 2.0,
) -> plt.Figure:
    """
    Plots a smoothed Precision-Recall (PR) curve per class and a global average.

    Args:
        per_true (Dict[int, List[int]]): Binary true labels per class.
        per_score (Dict[int, List[float]]): Prediction scores per class.
        labels_map (Dict[int, str]): Mapping from class index to label string.
        mAP (float): Mean Average Precision across all classes.
        sigma (float): Smoothing factor for curves.

    Returns:
        matplotlib.figure.Figure: PR curve figure.
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_title("Precision-Recall Curve", fontsize=13)

    classes = list(labels_map.keys())

    # Plot per-class PR curves
    for cls in classes:
        y_t = np.array(per_true[cls], dtype=int)
        y_s = np.array(per_score[cls], dtype=float)

        if y_t.sum() == 0:
            # Avoid division by zero: constant precision
            prec, rec = np.ones(10), np.linspace(0, 1, 10)
        else:
            prec, rec, _ = precision_recall_curve(y_t, y_s)

        # prec_s = smooth_curve(prec, sigma)
        # rec_s = smooth_curve(rec, sigma)
        ap = average_precision_score(y_t, y_s) if y_t.sum() > 0 else 0.0

        ax.plot(rec, prec, lw=2, label=f"{labels_map[cls]} {ap:.3f}")

    # Plot global PR curve
    all_true = np.concatenate([per_true[c] for c in classes])
    all_scores = np.concatenate([per_score[c] for c in classes])

    prec_all, rec_all, _ = precision_recall_curve(all_true, all_scores)
    # prec_all_s = smooth_curve(prec_all, sigma)
    # rec_all_s = smooth_curve(rec_all, sigma)
    ax.plot(
        rec_all,
        prec_all,
        lw=3,
        color="blue",
        label=f"all classes {mAP:.3f} mAP@0.5",
    )

    # Axes and styling
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)

    ax.legend(loc="upper left", bbox_to_anchor=(1.04, 1.0), fontsize=12, frameon=False)
    plt.tight_layout()
    print("[INFO] PR curve plotted.")
    return fig


def plot_confusion_matrix(
    y_true: List[int], y_pred: List[int], labels_map: Dict[int, str]
) -> Dict[str, plt.Figure]:
    """
    Plots both the raw and normalized confusion matrices.

    Args:
        y_true (List[int]): Ground truth class indices.
        y_pred (List[int]): Predicted class indices (may include -1 for background).
        labels_map (Dict[int, str]): Mapping from class index to class name.

    Returns:
        Dict[str, plt.Figure]: Dictionary with 'raw' and 'normalized' confusion matrix plots.
    """
    labels = list(labels_map.keys()) + [-1]
    names = [labels_map.get(l, "BG") for l in labels]

    cm_raw = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm_raw.astype(float) / cm_raw.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)  # Replace NaNs from division by zero

    # Raw matrix plot
    fig_raw, ax_raw = plt.subplots(figsize=(6, 6))
    im_raw = ax_raw.imshow(cm_raw, cmap="Blues")
    for i in range(len(names)):
        for j in range(len(names)):
            val = cm_raw[i, j]
            if val == 0:
                continue
            text = (
                f"{np.diag(cm_raw)[i]}/{cm_raw.sum(1)[i]}" if i == j else str(int(val))
            )
            color = "white" if val > cm_raw.max() / 2 else "black"
            ax_raw.text(j, i, text, ha="center", va="center", color=color)
    ax_raw.set_xticks(range(len(names)))
    ax_raw.set_yticks(range(len(names)))
    ax_raw.set_xticklabels(names, rotation=45, ha="right")
    ax_raw.set_yticklabels(names)
    ax_raw.set_xlabel("Predicted", fontsize=11)
    ax_raw.set_ylabel("True", fontsize=11)
    ax_raw.set_title("Confusion Matrix (Raw)", fontsize=13)
    plt.colorbar(im_raw, ax=ax_raw, fraction=0.046, pad=0.04)
    fig_raw.tight_layout()

    # Normalized matrix plot
    fig_norm, ax_norm = plt.subplots(figsize=(6, 6))
    im_norm = ax_norm.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)
    for i in range(len(names)):
        for j in range(len(names)):
            val = cm_norm[i, j]
            if val == 0:
                continue
            text = f"{val:.2f}"
            color = "white" if val > 0.5 else "black"
            ax_norm.text(j, i, text, ha="center", va="center", color=color)
    ax_norm.set_xticks(range(len(names)))
    ax_norm.set_yticks(range(len(names)))
    ax_norm.set_xticklabels(names, rotation=45, ha="right")
    ax_norm.set_yticklabels(names)
    ax_norm.set_xlabel("Predicted", fontsize=11)
    ax_norm.set_ylabel("True", fontsize=11)
    ax_norm.set_title("Confusion Matrix (Normalized)", fontsize=13)
    plt.colorbar(im_norm, ax=ax_norm, fraction=0.046, pad=0.04)
    fig_norm.tight_layout()

    print("[INFO] Confusion matrices plotted (raw and normalized).")
    return {"raw": fig_raw, "normalized": fig_norm}


def plot_child_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    figsize: Tuple[int, int] = (4, 4),
) -> Dict[str, plt.Figure]:
    """
    Plot the Adult (0) / Child (1) binary confusion matrix, returning both
    the raw counts and the row‑normalized version.

    Args:
        y_true (List[int]): Ground‑truth labels (0 = adult, 1 = child).
        y_pred (List[int]): Predicted labels  (0 = adult, 1 = child).
        figsize (Tuple[int, int]): Size of the output figures.

    Returns:
        Dict[str, plt.Figure]: A dict with keys **"raw"** and **"normalized"**
        mapping to the corresponding matplotlib figures.
    """
    # ------------------------------------------------------------------ #
    # 1) Compute raw and normalized matrices
    # ------------------------------------------------------------------ #
    cm_raw = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_norm = cm_raw.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm_norm, row_sums, where=row_sums != 0)

    classes = ["Adult", "Child"]

    # ------------------------------------------------------------------ #
    # 2) Plot raw confusion matrix
    # ------------------------------------------------------------------ #
    fig_raw, ax_raw = plt.subplots(figsize=figsize)
    im_raw = ax_raw.imshow(cm_raw, cmap="Blues")

    for i in range(2):
        for j in range(2):
            val = cm_raw[i, j]
            if val == 0:
                continue
            ax_raw.text(
                j,
                i,
                int(val),
                ha="center",
                va="center",
                color="white" if val > cm_raw.max() / 2 else "black",
            )

    ax_raw.set_xticks([0, 1])
    ax_raw.set_yticks([0, 1])
    ax_raw.set_xticklabels(classes)
    ax_raw.set_yticklabels(classes)
    ax_raw.set_xlabel("Predicted", fontsize=11)
    ax_raw.set_ylabel("True", fontsize=11)
    ax_raw.set_title("Adult / Child Confusion Matrix (Raw)", fontsize=13)
    plt.colorbar(im_raw, ax=ax_raw, fraction=0.046, pad=0.04)
    fig_raw.tight_layout()

    # ------------------------------------------------------------------ #
    # 3) Plot normalized confusion matrix
    # ------------------------------------------------------------------ #
    fig_norm, ax_norm = plt.subplots(figsize=figsize)
    im_norm = ax_norm.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)

    for i in range(2):
        for j in range(2):
            val = cm_norm[i, j]
            if val == 0:
                continue
            ax_norm.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                color="white" if val > 0.5 else "black",
            )

    ax_norm.set_xticks([0, 1])
    ax_norm.set_yticks([0, 1])
    ax_norm.set_xticklabels(classes)
    ax_norm.set_yticklabels(classes)
    ax_norm.set_xlabel("Predicted", fontsize=11)
    ax_norm.set_ylabel("True", fontsize=11)
    ax_norm.set_title("Adult / Child Confusion Matrix (Normalized)", fontsize=13)
    plt.colorbar(im_norm, ax=ax_norm, fraction=0.046, pad=0.04)
    fig_norm.tight_layout()

    print("[INFO] Adult/Child confusion matrices plotted.")
    return {"raw": fig_raw, "normalized": fig_norm}


def plot_boxplots(
    data: List[Dict[str, Any]],
    x_field: str,
    y_field: str,
    title: str,
    labels_map: Dict[int, str],
    y_lim: Tuple[float, float] = None,
    cmap_name: str = "tab10",
) -> plt.Figure:
    """
    Draws class-wise colored boxplots for any metric (IoU, angle error, etc.)
    and includes a legend with mean ± std per class.


    Args:
        data (List[Dict[str, Any]]): List of metric dictionaries with 'class' and value fields.
        x_field (str): Name of the field to group by (class).
        y_field (str): Metric name to plot.
        title (str): Plot title.
        labels_map (Dict[int, str]): Mapping from class index to label.
        y_lim (Tuple[float, float], optional): Y-axis limits.
        cmap_name (str): Name of the colormap to use.

    Returns:
        matplotlib.figure.Figure: Boxplot figure.
    """
    classes = list(labels_map.keys())
    class_names = [labels_map[c] for c in classes]

    # Organize values by class
    values = [
        [d[y_field] for d in data if d[x_field] == labels_map[c]] for c in classes
    ]

    # Compute mean ± std for each class
    mean_std_text = {}
    for i, val in enumerate(values):
        name = class_names[i]
        if val:
            mu = np.mean(val)
            sigma = np.std(val)
            mean_std_text[name] = f"{mu:.2f} ± {sigma:.2f}"
        else:
            mean_std_text[name] = "N/A"

    fig, ax = plt.subplots(figsize=(9, 6))

    # Basic boxplot (unstyled)
    bp = ax.boxplot(
        values,
        positions=np.arange(len(class_names)),
        notch=True,
        patch_artist=True,
        boxprops=dict(facecolor="none", edgecolor="black"),
        medianprops=dict(color="black"),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
    )

    # Apply colormap
    cmap = plt.get_cmap(cmap_name)
    colors = {}
    for i, box in enumerate(bp["boxes"]):
        this_color = cmap(i)
        colors[class_names[i]] = this_color
        box.set_facecolor(this_color)
        box.set_edgecolor("black")

    # Add jittered points
    for i, (name, val) in enumerate(zip(class_names, values)):
        if val:
            jittered_x = np.random.normal(i, 0.04, size=len(val))
            ax.scatter(
                jittered_x,
                val,
                alpha=0.7,
                edgecolors="black",
                color=colors[name],
                label=f"{name} {mean_std_text[name]}",
            )

    # Axes style
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel(y_field, fontsize=12)
    ax.set_title(title, fontsize=14)
    if y_lim:
        ax.set_ylim(y_lim)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    # Legend outside
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.04, 1.0),
        frameon=False,
        title=f"{y_field} per class",
    )

    plt.tight_layout()
    print(f"[INFO] Boxplot for '{y_field}' created.")
    return fig


def plot_f1_vs_threshold(
    all_gts: List[int],
    all_scores: List[float],
    all_preds: List[int],
    labels_map: Dict[int, str],
    default_th: float = 0.5,
    n_steps: int = 100,
    sigma: float = 2.0,
) -> plt.Figure:
    """
    Plots F1 Score vs. confidence threshold for each class, reconstructing
    predictions from all_scores and all_preds at each threshold.

    Args:
        all_gts      : List of true labels (integers).
        all_scores   : List of scores (float) associated with each prediction.
        all_preds    : List of originally predicted labels (but the previous threshold will be ignored).
        labels_map   : Dict[int, str] mapping index→class name.
        default_th   : “default” threshold (used only to show it as a reference).
        n_steps      : Number of equally spaced points in [0,1] to evaluate F1.
        sigma        : Smoothing factor for the curve.

    Returns:
        matplotlib.figure.Figure with F1 vs threshold curves per class.
    """
    thresholds = np.linspace(0.0, 1.0, n_steps)
    y_true = np.array(all_gts, dtype=int)
    classes = list(labels_map.keys())

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_title("F1 vs. Confidence Threshold", fontsize=13)

    for cls in classes:
        f1s = []
        for t in thresholds:
            # For each prediction: if score >= t, assign the original label;
            # if score < t, assign -1 (background)
            y_pred_t = [
                (lbl if sc >= t else -1) for sc, lbl in zip(all_scores, all_preds)
            ]
            # Compute F1 with zero_division=0
            f1_val = f1_score(
                y_true, y_pred_t, labels=classes, average=None, zero_division=0
            )
            f1s.append(f1_val[classes.index(cls)])

        f1s = np.array(f1s)
        f1_s = smooth_curve(f1s, sigma)
        ax.plot(thresholds, f1_s, lw=2, label=f"{labels_map[cls]} {f1_s.mean():.3f}")

        # Mark the optimal F1 point for this class
        best_i = f1_s.argmax()
        ax.axvline(
            thresholds[best_i],
            linestyle="--",
            lw=1,
            color=ax.get_lines()[-1].get_color(),
        )
        ax.scatter(
            [thresholds[best_i]],
            [f1_s[best_i]],
            s=50,
            zorder=3,
            color=ax.get_lines()[-1].get_color(),
        )

    # Reference to the default threshold
    ax.axvline(
        default_th,
        color="gray",
        linestyle=":",
        linewidth=1.0,
        label=f"default_th={default_th}",
    )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Confidence Threshold", fontsize=11)
    ax.set_ylabel("F1 Score", fontsize=11)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12)

    ax.legend(loc="upper left", bbox_to_anchor=(1.04, 1.0), fontsize=10, frameon=False)
    plt.tight_layout()
    print("[INFO] F1 vs. threshold curve plotted.")
    return fig


# -----------------------------------------------------------------------------
# IV. Qualitative Grid & Saving Individually
# -----------------------------------------------------------------------------


def plot_qualitative_grid(
    samples: List[
        Tuple[
            Any, Dict[str, torch.Tensor], str, torch.Tensor, torch.Tensor, torch.Tensor
        ]
    ],
    labels_map: Dict[int, str],
    grid_shape: Tuple[int, int],
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
) -> plt.Figure:
    """
    Creates a grid of sample predictions showing both ground truth and predicted oriented bounding boxes (OBBs).

    Args:
        samples (List[Tuple]): List of samples, where each sample contains:
            - image_tensor (torch.Tensor): Normalized image tensor
            - prediction_dict (Dict[str, torch.Tensor]): Model predictions including:
                - 'polygons': Vertex coordinates of predicted OBBs
                - 'labels': Predicted class labels
                - 'scores': Confidence scores
                - 'boxes': OBB parameters (x,y,w,h,θ)
            - filename (str): Original image filename
            - gt_boxes (torch.Tensor): Ground truth OBB vertex coordinates
            - gt_angles (torch.Tensor): Ground truth rotation angles
            - gt_labels (torch.Tensor): Ground truth class labels
            - fp_count (int): Number of false positives
            - fn_count (int): Number of false negatives
            - viz_payload (Optional[Dict]): Optional visualization metadata
        labels_map (Dict[int, str]): Mapping from class indices to human-readable labels
        grid_shape (Tuple[int, int]): Number of (rows, columns) in the visualization grid
        mean (Tuple[float, float, float]): Channel-wise means for image denormalization
        std (Tuple[float, float, float]): Channel-wise standard deviations for denormalization

    Returns:
        matplotlib.figure.Figure: Figure containing the grid of visualizations with both
        ground truth (green dashed) and predicted (blue solid) oriented bounding boxes,
        each annotated with class label, angle and confidence score.
    """
    rows, cols = grid_shape
    # Create figure with white background for better visualization
    fig, axes = plt.subplots(
        rows, cols, figsize=(cols * 4, rows * 4), facecolor="white"
    )
    axes = axes.flatten()

    # Process only enough samples to fill the grid
    for ax, sample in zip(axes, samples[: rows * cols]):
        # Handle both 8-element and 9-element sample tuples (with/without viz_payload)
        if len(sample) == 9:
            img_t, out, fname, gt_b, gt_a, gt_l, fp_img, fn_img, _viz = sample
        else:
            img_t, out, fname, gt_b, gt_a, gt_l, fp_img, fn_img = sample

        # Display denormalized image and configure axis
        ax.imshow(denormalize_image(img_t, mean=mean, std=std))
        ax.axis("off")
        ax.set_title(f"{Path(fname).name}\nFP:{fp_img}  FN:{fn_img}", fontsize=7)
        ax.set_aspect("equal")

        # Draw ground truth OBBs (green dashed boxes)
        for pts, angle, cls in zip(gt_b, gt_a, gt_l):
            pts_np = pts.view(4, 2).numpy()
            # Draw OBB polygon
            ax.add_patch(
                patches.Polygon(
                    pts_np,
                    closed=True,
                    fill=False,
                    edgecolor="#008000",  # Dark green
                    linewidth=2,
                    linestyle="--",
                )
            )
            # Draw front edge (orientation indicator)
            ax.plot(pts_np[[0, 1], 0], pts_np[[0, 1], 1], color="orange", linewidth=2)

            # Add label with class and angle at bottom-right
            br_x, br_y = pts_np[:, 0].max(), pts_np[:, 1].max()
            ax.text(
                br_x,
                br_y,
                f"{labels_map.get(int(cls), 'unknown')}: {math.degrees(float(angle)):.1f}°",
                color="white",
                fontsize=6,
                fontweight="bold",
                ha="right",
                va="bottom",
                bbox=dict(facecolor="#008000", alpha=0.8, edgecolor="none", pad=2.5),
            )

        # Draw predicted OBBs (blue solid boxes)
        for i, (pts, lbl, score) in enumerate(
            zip(out["polygons"], out["labels"], out["final_score"])
        ):
            pts_np = pts.view(4, 2).numpy()
            # Draw OBB polygon
            ax.add_patch(
                patches.Polygon(
                    pts_np,
                    closed=True,
                    fill=False,
                    edgecolor="#004080",  # Dark blue
                    linewidth=1.5,
                )
            )
            # Draw front edge (orientation indicator)
            ax.plot(
                pts_np[[0, 1], 0], pts_np[[0, 1], 1], color="#800000", linewidth=1.5
            )

            # Add label with class, angle and score at top-left
            tl_x, tl_y = pts_np[:, 0].min(), pts_np[:, 1].min()
            ang = math.degrees(float(out["boxes"][i, 4]))
            ax.text(
                tl_x,
                tl_y,
                f"{labels_map.get(int(lbl), 'unknown')}: {ang:.1f}° / {score:.2f}",
                color="white",
                fontsize=6,
                ha="left",
                va="top",
                bbox=dict(facecolor="#004080", alpha=0.9, edgecolor="none", pad=2.5),
            )

    # Hide any unused axes in the grid
    for ax in axes[len(samples) :]:
        ax.axis("off")

    fig.tight_layout(pad=0.5)
    print("[INFO] Grid of qualitative predictions plotted.")
    return fig


def plot_histograms_split(
    data: Dict[str, Any],
    labels_map: Dict[int, str],
    bin_deg: int,
    out_dir: Path,
    tag: str,
) -> None:
    """
    Plot and save histograms of GT angles (degrees) for all samples and per class.

    Args:
        data: Output of `collect_degrees_by_class`.
        labels_map: Mapping class_idx -> class name.
        bin_deg: Histogram bin width in degrees.
        out_dir: Output directory for images.
        tag: Prefix tag for output filenames and titles.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    bins = np.arange(0, 180 + bin_deg, bin_deg)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(data["all"], bins=bins, edgecolor="black")
    ax.set_title(f"{tag}: GT angle histogram (ALL), bin={bin_deg} deg")
    ax.set_xlabel("GT angle [deg]")
    ax.set_ylabel("Count")
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / f"{tag}_ALL_bin{bin_deg}.png", dpi=200)
    plt.close(fig)

    classes = list(labels_map.keys())
    n_cls = len(classes)
    n_cols = min(3, n_cls)
    n_rows = int(math.ceil(n_cls / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.8 * n_rows))
    axes = np.atleast_2d(axes)

    for i, c in enumerate(classes):
        r, col = divmod(i, n_cols)
        ax = axes[r, col]
        values = data["per_cls"][c]
        ax.hist(values, bins=bins, edgecolor="black")
        ax.set_title(f"{labels_map[c]} (n={len(values)}), bin={bin_deg} deg")
        ax.set_xlabel("GT angle [deg]")
        ax.set_ylabel("Count")
        ax.grid(axis="y", linestyle=":", alpha=0.6)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    for k in range(n_cls, n_rows * n_cols):
        r, col = divmod(k, n_cols)
        axes[r, col].axis("off")

    fig.suptitle(f"{tag}: GT angle histogram per class, bin={bin_deg} deg")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_dir / f"{tag}_perclass_bin{bin_deg}.png", dpi=200)
    plt.close(fig)
