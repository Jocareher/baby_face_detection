# build_min_bbox_from_landmarks_simple.py
# -*- coding: utf-8 -*-

"""
Compute minimal axis-aligned bounding boxes from 68-point landmark files.

Assumed dataset structure:
    ROOT/
      images/*.png (or jpg/jpeg/bmp/tif/tiff/webp)
      labels/*.txt  (each with 68 landmarks)

Outputs:
    OUTPUT_DIR/
      labels/<stem>.txt  -> "x1 y1 x2 y2"
      plots/<filename>   -> original image with bbox overlay

Usage:
    python build_min_bbox_from_landmarks_simple.py \
        --root /path/to/dataset \
        --output-dir /path/to/output \
        --padding 0.05
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass  # For simple data structures
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


SUPPORTED_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class BBox:
    """Axis-aligned bounding box in absolute pixel coordinates."""

    x1: int
    y1: int
    x2: int
    y2: int

    def clamp(self, w: int, h: int) -> "BBox":
        """Clamp the bbox to image bounds and ensure >= 1×1 size."""
        x1 = int(np.clip(self.x1, 0, max(0, w - 1)))
        y1 = int(np.clip(self.y1, 0, max(0, h - 1)))
        x2 = int(np.clip(self.x2, 0, max(0, w - 1)))
        y2 = int(np.clip(self.y2, 0, max(0, h - 1)))
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        if x2 == x1 and w > 1:
            x2 = min(x1 + 1, w - 1)
        if y2 == y1 and h > 1:
            y2 = min(y1 + 1, h - 1)
        return BBox(x1, y1, x2, y2)

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2, self.y2


def parse_landmarks(txt_path: Path) -> np.ndarray:
    """
    Parse a 68-point landmark file into (N, 2) float32 array.

    Accepts:
      - 68 lines: "x y" (spaces or commas)
      - Single line with 136 numbers: "x1 y1 x2 y2 ... x68 y68"
    """
    raw = txt_path.read_text().strip()
    if not raw:
        raise ValueError(f"Empty label file: {txt_path}")

    tokens = raw.replace(",", " ").split()
    vals = [float(t) for t in tokens]
    if len(vals) % 2 != 0:
        raise ValueError(f"Odd count in {txt_path}; expected pairs.")
    pts = np.array(vals, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] < 4:
        raise ValueError(f"Too few points in {txt_path}: {pts.shape[0]}")
    return pts


def compute_min_bbox(
    landmarks: np.ndarray,
    img_w: int,
    img_h: int,
    padding: float = 0.0,
) -> BBox:
    """
    Compute minimal axis-aligned bbox covering all landmarks, with optional padding (fraction).
    """
    xs, ys = landmarks[:, 0], landmarks[:, 1]
    x_min, x_max = float(np.min(xs)), float(np.max(xs))
    y_min, y_max = float(np.min(ys)), float(np.max(ys))

    bw = max(1.0, x_max - x_min)
    bh = max(1.0, y_max - y_min)

    if padding > 1e-9:
        x_min -= bw * padding
        x_max += bw * padding
        y_min -= bh * padding
        y_max += bh * padding

    bbox = BBox(
        int(np.floor(x_min)),
        int(np.floor(y_min)),
        int(np.ceil(x_max)),
        int(np.ceil(y_max)),
    )
    return bbox.clamp(img_w, img_h)


def draw_bbox(
    image_bgr: np.ndarray, bbox: BBox, color=(0, 255, 0), thickness: int | None = None
) -> np.ndarray:
    """Draw rectangle on a BGR image and return a copy."""
    out = image_bgr.copy()
    h, w = out.shape[:2]
    if thickness is None:
        thickness = max(2, int(0.002 * (h + w)))
    cv2.rectangle(
        out, (bbox.x1, bbox.y1), (bbox.x2, bbox.y2), color, thickness, cv2.LINE_AA
    )
    return out


def process_dataset(
    root: Path,
    output_dir: Path,
    padding: float = 0.0,
) -> None:
    """
    Process ROOT/images and ROOT/labels, writing results to OUTPUT_DIR.

    Writes:
      - OUTPUT_DIR/labels/<stem>.txt  with "x1 y1 x2 y2"
      - OUTPUT_DIR/plots/<filename>   original image with bbox overlay
    """
    img_dir = root / "images"
    lbl_dir = root / "labels"

    # Create outputs
    out_labels = output_dir / "labels"
    out_plots = output_dir / "plots"
    out_labels.mkdir(parents=True, exist_ok=True)
    out_plots.mkdir(parents=True, exist_ok=True)

    # Gather images
    img_paths: List[Path] = sorted(
        [p for p in img_dir.rglob("*") if p.suffix.lower() in SUPPORTED_IMG_EXTS]
    )
    if not img_paths:
        print(f"[WARN] No images found under: {img_dir}")
        return

    print(f"Found {len(img_paths)} images under {img_dir}")
    missing, processed = 0, 0

    for img_path in img_paths:
        stem = img_path.stem
        lbl_path = lbl_dir / f"{stem}.txt"

        if not lbl_path.exists():
            missing += 1
            print(f"[WARN] Missing label for: {img_path.name}")
            continue

        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"[WARN] Could not read image: {img_path}")
            continue
        h, w = img_bgr.shape[:2]

        try:
            lms = parse_landmarks(lbl_path)  # (N, 2)
        except Exception as e:
            print(f"[WARN] Failed to parse {lbl_path}: {e}")
            continue

        bbox = compute_min_bbox(lms, w, h, padding=padding)

        # Save bbox txt
        with open(out_labels / f"{stem}.txt", "w") as f:
            f.write(f"{bbox.x1} {bbox.y1} {bbox.x2} {bbox.y2}\n")

        # Save plot
        vis = draw_bbox(img_bgr, bbox)
        cv2.imwrite(str(out_plots / img_path.name), vis)

        processed += 1
        if processed % 100 == 0:
            print(f"[{processed}/{len(img_paths)}] processed...")

    print(f"Done. Processed {processed}. Missing labels: {missing}.")
    print(f"Labels -> {out_labels}")
    print(f"Plots  -> {out_plots}")


def build_argparser() -> argparse.ArgumentParser:
    """CLI parser: only needs root and output_dir (plus optional padding)."""
    ap = argparse.ArgumentParser(
        description="Compute min AABB from 68-point landmarks (ROOT/{images,labels})."
    )
    ap.add_argument(
        "--root",
        type=str,
        required=True,
        help="Dataset root directory (contains images/ and labels/).",
    )
    ap.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory (will contain labels/ and plots/).",
    )
    ap.add_argument(
        "--padding",
        type=float,
        default=0.0,
        help="Padding ratio (e.g., 0.05 adds 5% on each side).",
    )
    return ap


def main() -> None:
    args = build_argparser().parse_args()
    process_dataset(
        root=Path(args.root),
        output_dir=Path(args.output_dir),
        padding=args.padding,
    )


if __name__ == "__main__":
    main()
