"""
Rotate baby images in a dataset using model-predicted angles, update OBB labels, and
export a new dataset with expanded canvas and visualizations.

Behavior:
- BG (no .txt): copy image only (no label created), optional vis not generated.
- ADULT_ONLY (no baby GT): copy image + original labels unchanged.
- BABY:
  - If number of baby GT != 1: copy unchanged (image + labels).
  - Else:
      - Run inference, scale predictions to original image coordinates.
      - Match the single baby GT to best predicted box by pIoU.
      - If best IoU < match_thr: copy unchanged.
      - Else:
          - Rotate image by rot_angle = -theta_pred (CCW+ convention).
          - Transform all GT polygons with same affine.
          - Update each GT angle: wrap_to_pi(wrap_to_pi(theta_gt) + rot_angle)
          - Normalize polygons w.r.t. new image size.
          - Save rotated image and updated labels.
          - Save visualization image with polygons drawn.

Important conventions:
- Model predicted theta is assumed CCW positive, consistent with cos/sin usage in decode.
- We rotate the image by -theta_pred to "deskew" the matched baby face.
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import your project modules here.
# Adjust these imports to match your repository structure.
from models.newborn import NewBORN
from engine.inference import infer_with_rotated_nms
from utils.helpers import set_seed
from loss.utils import batch_probiou, xyxyxyxy2xywhr
import config


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments.

    Returns:
        Parsed arguments namespace.
    """
    p = argparse.ArgumentParser("Rotate baby dataset using model predictions.")
    p.add_argument("--root_dir", type=str, required=True, help="Dataset root.")
    p.add_argument(
        "--split", type=str, default="train", help="Split name (train/val/test)."
    )
    p.add_argument(
        "--output_root", type=str, required=True, help="Output dataset root."
    )
    p.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint."
    )
    p.add_argument(
        "--backbone",
        type=str,
        default="densenet121",
        choices=["mobilenetv1", "resnet50", "vgg16", "densenet121", "vit", "vggface2"],
    )
    p.add_argument("--out_channel", type=int, default=config.DEFAULT_OUT_CHANNELS)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)

    p.add_argument("--face_thres", type=float, default=config.FACE_THRESH)
    p.add_argument("--baby_thres", type=float, default=config.BABY_THRESH)
    p.add_argument("--class_thres", type=float, default=config.CLASS_THRESH)
    p.add_argument("--nms_iou_thres", type=float, default=config.IOU_THRESH)

    p.add_argument(
        "--match_iou_thr",
        type=float,
        default=0.50,
        help="Minimum pIoU required to rotate an image (matching GT to a prediction).",
    )
    p.add_argument(
        "--border_value",
        type=int,
        nargs=3,
        default=[0, 0, 0],
        help="Zero padding color as three ints (B G R). Default is 0 0 0.",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def get_default_device() -> torch.device:
    """
    Get the best available device.

    Returns:
        Torch device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if (
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")
    return torch.device("cpu")


def wrap_to_pi_float(angle: float) -> float:
    """
    Wrap an angle in radians to [-pi, pi].

    Args:
        angle: Angle in radians.

    Returns:
        Wrapped angle.
    """
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


@dataclass
class SamplePaths:
    """
    Container for per-image paths in the dataset.
    """

    image_path: Path
    label_path: Path


def list_dataset_samples(root_dir: Path, split: str) -> List[SamplePaths]:
    """
    List all image samples for a given split.

    Args:
        root_dir: Dataset root directory.
        split: Split name.

    Returns:
        List of SamplePaths for all .jpg images found in split/images.
    """
    images_dir = root_dir / split / "images"
    labels_dir = root_dir / split / "labels"

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    samples: List[SamplePaths] = []
    for p in sorted(images_dir.rglob("*.jpg")):
        stem = p.stem
        lbl = labels_dir / f"{stem}.txt"
        samples.append(SamplePaths(image_path=p, label_path=lbl))
    return samples


def read_raw_gt_lines(path: Path) -> List[str]:
    """
    Read all non-empty lines from a GT file.

    Args:
        path: Label file path.

    Returns:
        List of non-empty lines. Empty if file doesn't exist.
    """
    if not path.exists():
        return []
    with path.open("r") as f:
        return [ln.strip() for ln in f if ln.strip()]


def classify_image_gt(gt_path: Path) -> str:
    """
    Classify GT file as BABY, ADULT_ONLY, or BG.

    BABY means at least one line with class_idx != -1 (your baby orientations 0..4).
    ADULT_ONLY means only class_idx == -1 lines.
    BG means missing or empty or invalid.

    Args:
        gt_path: Label file path.

    Returns:
        "BABY", "ADULT_ONLY", or "BG".
    """
    lines = read_raw_gt_lines(gt_path)
    if not lines:
        return "BG"

    has_baby = False
    has_any = False
    has_adult = False

    for ln in lines:
        toks = ln.split()
        if not toks:
            continue
        try:
            cls_idx = int(float(toks[0]))
        except Exception:
            continue
        has_any = True
        if cls_idx != -1:
            has_baby = True
        if cls_idx == -1:
            has_adult = True

    if not has_any:
        return "BG"
    if has_baby:
        return "BABY"
    if has_adult:
        return "ADULT_ONLY"
    return "BG"


@dataclass
class GTEntry:
    """
    Parsed GT entry with normalized polygon and angle.
    """

    cls_idx: int
    child_prob: int
    poly_norm_42: np.ndarray  # (4,2) float32 in [0,1]
    angle_rad: float


def parse_gt_entries(gt_path: Path) -> List[GTEntry]:
    """
    Parse your label format:
        class_idx child_prob x1 y1 x2 y2 x3 y3 x4 y4 angle

    Args:
        gt_path: Path to GT file.

    Returns:
        List of parsed GT entries.
    """
    lines = read_raw_gt_lines(gt_path)
    out: List[GTEntry] = []
    for ln in lines:
        toks = ln.split()
        if len(toks) != 11:
            continue
        try:
            cls_idx = int(float(toks[0]))
            child_prob = int(float(toks[1]))
            coords = np.array([float(x) for x in toks[2:10]], dtype=np.float32).reshape(
                4, 2
            )
            angle = float(toks[10])
        except Exception:
            continue
        out.append(
            GTEntry(
                cls_idx=cls_idx,
                child_prob=child_prob,
                poly_norm_42=coords,
                angle_rad=angle,
            )
        )
    return out


def poly_norm_to_pix(poly_norm_42: np.ndarray, w: int, h: int) -> np.ndarray:
    """
    Convert normalized polygon to pixel coordinates.

    Args:
        poly_norm_42: (4,2) normalized.
        w: Image width.
        h: Image height.

    Returns:
        (4,2) polygon in pixels.
    """
    p = poly_norm_42.astype(np.float32).copy()
    p[:, 0] *= float(w)
    p[:, 1] *= float(h)
    return p


def poly_pix_to_norm(poly_pix_42: np.ndarray, w: int, h: int) -> np.ndarray:
    """
    Convert pixel polygon to normalized coordinates.

    Args:
        poly_pix_42: (4,2) pixels.
        w: Image width.
        h: Image height.

    Returns:
        (4,2) normalized.
    """
    p = poly_pix_42.astype(np.float32).copy()
    p[:, 0] /= float(w)
    p[:, 1] /= float(h)
    return p


@dataclass
class RotationResult:
    """
    Result of an expanded-canvas rotation.
    """

    rotated_image_bgr: np.ndarray
    rotated_polys_42: np.ndarray
    affine_2x3: np.ndarray
    new_size_wh: Tuple[int, int]


def rotate_expand_and_transform_polys(
    image_bgr: np.ndarray,
    polys_42: np.ndarray,
    angle_rad_ccw: float,
    border_value_bgr: Tuple[int, int, int],
) -> RotationResult:
    """
    Rotate full image by angle (CCW+), expand canvas, and transform polygons.

    Args:
        image_bgr: HxWxC image in BGR.
        polys_42: (N,4,2) polygons in pixel coordinates.
        angle_rad_ccw: Rotation angle in radians (CCW positive).
        border_value_bgr: Constant padding color (B,G,R).

    Returns:
        RotationResult.
    """
    h, w = image_bgr.shape[:2]
    cx, cy = (w - 1) * 0.5, (h - 1) * 0.5
    angle_deg = -float(math.degrees(angle_rad_ccw))

    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0).astype(np.float32)

    corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], dtype=np.float32).T
    rot_xy = (M @ corners).T[:, :2]
    min_xy = rot_xy.min(axis=0)
    max_xy = rot_xy.max(axis=0)

    tx, ty = -min_xy[0], -min_xy[1]
    M[:, 2] += np.array([tx, ty], dtype=np.float32)

    new_w = int(math.ceil(max_xy[0] - min_xy[0]))
    new_h = int(math.ceil(max_xy[1] - min_xy[1]))

    rotated = cv2.warpAffine(
        image_bgr,
        M,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value_bgr,
    )

    if polys_42.size == 0:
        rot_polys = polys_42.reshape(0, 4, 2).astype(np.float32)
    else:
        N = polys_42.shape[0]
        poly_h = np.concatenate(
            [polys_42.astype(np.float32), np.ones((N, 4, 1), dtype=np.float32)],
            axis=2,
        )  # (N,4,3)
        rot_polys = (M[None, :, :] @ poly_h.transpose(0, 2, 1)).transpose(0, 2, 1)[
            :, :, :2
        ]

    return RotationResult(
        rotated_image_bgr=rotated,
        rotated_polys_42=rot_polys.astype(np.float32),
        affine_2x3=M,
        new_size_wh=(new_w, new_h),
    )


def draw_polygons_bgr(
    image_bgr: np.ndarray,
    polys_42: np.ndarray,
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw polygons on an image for visualization.

    Args:
        image_bgr: Image in BGR.
        polys_42: (N,4,2) polygons in pixel coords.
        thickness: Line thickness.

    Returns:
        Copy of the image with polygons drawn.
    """
    out = image_bgr.copy()
    if polys_42 is None or polys_42.size == 0:
        return out

    polys_int = np.round(polys_42).astype(np.int32)
    for p in polys_int:
        cv2.polylines(
            out,
            [p.reshape(-1, 1, 2)],
            isClosed=True,
            color=(0, 255, 0),
            thickness=thickness,
        )
    return out


def load_checkpoint_into_model(
    model: nn.Module, checkpoint_path: Path, device: torch.device
) -> None:
    """
    Load a checkpoint with robust key handling.

    Args:
        model: Model instance.
        checkpoint_path: Path to checkpoint.
        device: Target device.
    """
    raw = torch.load(str(checkpoint_path), map_location=device)
    state = raw.get("model_state_dict", raw)

    if any(k.startswith("_orig_mod.") for k in state):
        state = {
            (k[len("_orig_mod.") :] if k.startswith("_orig_mod.") else k): v
            for k, v in state.items()
        }

    model.load_state_dict(state, strict=True)


def build_model(args: argparse.Namespace, device: torch.device) -> nn.Module:
    """
    Build and load the NewBORN model.

    Args:
        args: CLI arguments.
        device: Device.

    Returns:
        Loaded model in eval mode.
    """
    model = NewBORN(
        backbone_name=args.backbone, out_channel=args.out_channel, pretrained=False
    ).to(device)
    load_checkpoint_into_model(model, Path(args.checkpoint), device)
    model.eval()
    return model


def preprocess_for_model_rgb(
    image_bgr: np.ndarray,
    resize_wh: Tuple[int, int],
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
) -> torch.Tensor:
    """
    Convert a BGR uint8 image to a normalized tensor for the model.

    Args:
        image_bgr: HxWxC BGR uint8 image.
        resize_wh: (W,H) resize.
        mean: RGB mean.
        std: RGB std.

    Returns:
        Tensor of shape (3,H,W) float32.
    """
    wr, hr = resize_wh
    resized = cv2.resize(image_bgr, (wr, hr), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
    mean_t = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
    std_t = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
    return (x - mean_t) / std_t


def write_labels_normalized(
    out_path: Path,
    entries: List[GTEntry],
) -> None:
    """
    Write labels in the original GT format with normalized coords.

    Args:
        out_path: Destination .txt path.
        entries: Entries to write.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for e in entries:
            p = e.poly_norm_42.reshape(-1)
            f.write(
                f"{e.cls_idx} {e.child_prob} "
                f"{p[0]:.9f} {p[1]:.9f} {p[2]:.9f} {p[3]:.9f} "
                f"{p[4]:.9f} {p[5]:.9f} {p[6]:.9f} {p[7]:.9f} "
                f"{e.angle_rad:.9f}\n"
            )


def ensure_dirs(out_root: Path, split: str) -> Tuple[Path, Path, Path]:
    """
    Create output directories for images, labels, and visualizations.

    Args:
        out_root: Output root.
        split: Split name.

    Returns:
        (images_dir, labels_dir, vis_dir)
    """
    images_dir = out_root / split / "images"
    labels_dir = out_root / split / "labels"
    vis_dir = out_root / split / "vis"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)
    return images_dir, labels_dir, vis_dir


def copy_image_and_labels(
    src_img: Path,
    src_lbl: Path,
    dst_img: Path,
    dst_lbl: Path,
    write_empty_label_if_missing: bool = False,
) -> None:
    """
    Copy image and label file to destination.

    Args:
        src_img: Source image path.
        src_lbl: Source label path.
        dst_img: Destination image path.
        dst_lbl: Destination label path.
        write_empty_label_if_missing: If True, create empty .txt when src label missing.
    """
    dst_img.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_img, dst_img)

    if src_lbl.exists():
        dst_lbl.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_lbl, dst_lbl)
    else:
        if write_empty_label_if_missing:
            dst_lbl.parent.mkdir(parents=True, exist_ok=True)
            dst_lbl.write_text("")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_default_device()

    root_dir = Path(args.root_dir)
    out_root = Path(args.output_root)
    split = args.split

    resize_size = list(config.PRECOMPUTED_OBB_STATS.keys())[0]  # (W,H)
    wr, hr = resize_size

    images_out_dir, labels_out_dir, vis_out_dir = ensure_dirs(out_root, split)

    model = build_model(args, device)

    anchors_xy = torch.load(config.ANCHORS_CACHE_PATH, map_location="cpu")[
        "anchors_xy"
    ].to(device)
    nms_image_size_hw = (hr, wr)

    samples = list_dataset_samples(root_dir, split)

    border_value = tuple(int(x) for x in args.border_value)

    processed = 0
    rotated = 0
    copied = 0
    skipped_multi = 0
    no_match = 0
    no_pred = 0

    for s in samples:
        processed += 1
        img_path = s.image_path
        lbl_path = s.label_path
        stem = img_path.stem

        dst_img = images_out_dir / f"{stem}.jpg"
        dst_lbl = labels_out_dir / f"{stem}.txt"
        dst_vis = vis_out_dir / f"{stem}.jpg"

        gt_kind = classify_image_gt(lbl_path)

        # BG: no label file, copy only image, skip visualization
        if gt_kind == "BG":
            copy_image_and_labels(
                img_path, lbl_path, dst_img, dst_lbl, write_empty_label_if_missing=False
            )
            copied += 1
            continue

        # Load image (BGR)
        image_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        h0, w0 = image_bgr.shape[:2]

        # Parse labels and keep as-is for adults
        gt_entries = parse_gt_entries(lbl_path)

        if gt_kind == "ADULT_ONLY":
            copy_image_and_labels(
                img_path, lbl_path, dst_img, dst_lbl, write_empty_label_if_missing=True
            )
            # Visualization for adult-only as well
            polys_pix = []
            for e in gt_entries:
                polys_pix.append(poly_norm_to_pix(e.poly_norm_42, w0, h0))
            polys_pix_np = (
                np.stack(polys_pix, axis=0).astype(np.float32)
                if polys_pix
                else np.zeros((0, 4, 2), np.float32)
            )
            vis = draw_polygons_bgr(image_bgr, polys_pix_np)
            cv2.imwrite(str(dst_vis), vis)
            copied += 1
            continue

        # BABY case
        baby_entries = [
            e for e in gt_entries if e.child_prob == 1 and (0 <= e.cls_idx <= 4)
        ]
        if len(baby_entries) != 1:
            copy_image_and_labels(
                img_path, lbl_path, dst_img, dst_lbl, write_empty_label_if_missing=True
            )
            polys_pix = []
            for e in gt_entries:
                polys_pix.append(poly_norm_to_pix(e.poly_norm_42, w0, h0))
            polys_pix_np = (
                np.stack(polys_pix, axis=0).astype(np.float32)
                if polys_pix
                else np.zeros((0, 4, 2), np.float32)
            )
            vis = draw_polygons_bgr(image_bgr, polys_pix_np)
            cv2.imwrite(str(dst_vis), vis)
            copied += 1
            skipped_multi += 1
            continue

        # Prepare model input
        x = (
            preprocess_for_model_rgb(
                image_bgr=image_bgr,
                resize_wh=(wr, hr),
                mean=config.IMAGENET_MEAN,
                std=config.IMAGENET_STD,
            )
            .unsqueeze(0)
            .to(device)
        )

        # Inference
        outputs = infer_with_rotated_nms(
            model_or_preds=model,
            images=x,
            anchors_xy=anchors_xy,
            image_size=nms_image_size_hw,
            face_thres=args.face_thres,
            baby_thres=args.baby_thres,
            iou_thres=args.nms_iou_thres,
            class_thres=args.class_thres,
        )

        out0 = outputs[0]
        pred_boxes = out0.get("boxes")  # (N,5) in resized coords (cx,cy,w,h,theta)
        if pred_boxes is None or pred_boxes.numel() == 0:
            copy_image_and_labels(
                img_path, lbl_path, dst_img, dst_lbl, write_empty_label_if_missing=True
            )
            polys_pix = []
            for e in gt_entries:
                polys_pix.append(poly_norm_to_pix(e.poly_norm_42, w0, h0))
            polys_pix_np = (
                np.stack(polys_pix, axis=0).astype(np.float32)
                if polys_pix
                else np.zeros((0, 4, 2), np.float32)
            )
            vis = draw_polygons_bgr(image_bgr, polys_pix_np)
            cv2.imwrite(str(dst_vis), vis)
            copied += 1
            no_pred += 1
            continue

        # Scale predicted xywhr to original
        sx = float(w0) / float(wr)
        sy = float(h0) / float(hr)
        pred_boxes_np = pred_boxes.detach().cpu().numpy().astype(np.float32)
        pred_boxes_np[:, 0] *= sx
        pred_boxes_np[:, 1] *= sy
        pred_boxes_np[:, 2] *= sx
        pred_boxes_np[:, 3] *= sy
        pred_xywhr = torch.from_numpy(pred_boxes_np).to(device)

        # Build GT baby xywhr (original coords)
        baby = baby_entries[0]
        baby_poly_pix = poly_norm_to_pix(baby.poly_norm_42, w0, h0).reshape(1, 4, 2)
        baby_angle_wrapped = wrap_to_pi_float(float(baby.angle_rad))
        gt_boxes_xy = torch.from_numpy(
            baby_poly_pix.reshape(1, 8).astype(np.float32)
        ).to(device)
        gt_angles = torch.tensor(
            [baby_angle_wrapped], dtype=torch.float32, device=device
        )

        # Convert GT to xywhr via your helper
        gt_xywhr = xyxyxyxy2xywhr(gt_boxes_xy, gt_angles, (w0, h0))

        # Match by pIoU
        ious = batch_probiou(pred_xywhr, gt_xywhr)  # (N,1) if implemented as (N,M)
        if ious.ndim == 2:
            ious_1d = ious[:, 0]
        else:
            ious_1d = ious

        best_iou, best_idx = torch.max(ious_1d, dim=0)
        if float(best_iou.item()) < float(args.match_iou_thr):
            copy_image_and_labels(
                img_path, lbl_path, dst_img, dst_lbl, write_empty_label_if_missing=True
            )
            polys_pix = []
            for e in gt_entries:
                polys_pix.append(poly_norm_to_pix(e.poly_norm_42, w0, h0))
            polys_pix_np = (
                np.stack(polys_pix, axis=0).astype(np.float32)
                if polys_pix
                else np.zeros((0, 4, 2), np.float32)
            )
            vis = draw_polygons_bgr(image_bgr, polys_pix_np)
            cv2.imwrite(str(dst_vis), vis)
            copied += 1
            no_match += 1
            continue

        theta_pred = float(pred_boxes_np[int(best_idx.item()), 4])
        rot_angle = -theta_pred  # deskew

        # Transform ALL GT polys (including adult lines if any exist)
        all_polys_pix = []
        all_angles_wrapped = []
        for e in gt_entries:
            poly_pix = poly_norm_to_pix(e.poly_norm_42, w0, h0)
            all_polys_pix.append(poly_pix)
            all_angles_wrapped.append(wrap_to_pi_float(float(e.angle_rad)))

        all_polys_pix_np = (
            np.stack(all_polys_pix, axis=0).astype(np.float32)
            if all_polys_pix
            else np.zeros((0, 4, 2), np.float32)
        )

        rot_res = rotate_expand_and_transform_polys(
            image_bgr=image_bgr,
            polys_42=all_polys_pix_np,
            angle_rad_ccw=rot_angle,
            border_value_bgr=border_value,
        )
        new_w, new_h = rot_res.new_size_wh

        # Update label entries: new polys normalized, new angles wrapped
        updated_entries: List[GTEntry] = []
        for i, e in enumerate(gt_entries):
            poly_rot_pix = rot_res.rotated_polys_42[i]
            poly_rot_norm = poly_pix_to_norm(poly_rot_pix, new_w, new_h)

            angle_new = wrap_to_pi_float(all_angles_wrapped[i] + rot_angle)

            updated_entries.append(
                GTEntry(
                    cls_idx=e.cls_idx,
                    child_prob=e.child_prob,
                    poly_norm_42=poly_rot_norm,
                    angle_rad=angle_new,
                )
            )

        # Save rotated image and updated labels
        cv2.imwrite(str(dst_img), rot_res.rotated_image_bgr)
        write_labels_normalized(dst_lbl, updated_entries)

        # Save visualization on rotated image
        vis = draw_polygons_bgr(rot_res.rotated_image_bgr, rot_res.rotated_polys_42)
        cv2.imwrite(str(dst_vis), vis)

        rotated += 1

    print("Done.")
    print(f"Processed: {processed}")
    print(f"Rotated  : {rotated}")
    print(f"Copied   : {copied}")
    print(f"Skipped multi-baby: {skipped_multi}")
    print(f"No pred  : {no_pred}")
    print(f"No match : {no_match}")
    print(f"Output images: {images_out_dir}")
    print(f"Output labels: {labels_out_dir}")
    print(f"Output vis   : {vis_out_dir}")


if __name__ == "__main__":
    main()
