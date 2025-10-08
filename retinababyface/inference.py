import argparse
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import torch
from torch.utils.data import DataLoader

from data_setup.dataset import BabyFacesDataset
from data_setup.collate import custom_collate
from models.retinababyface import RetinaBabyFace
from engine.inference import (
    infer_with_rotated_nms,
    load_original_and_scale,
    denormalize_image,
)
from utils.helpers import (
    get_default_device,
    seed_worker,
    set_seed,
    to_numpy,
    ensure_polygons_42_shape,
)
from utils.visualize import (
    draw_predictions_on_image,
    write_predictions_txt,
    xywhr_to_poly42_shape,
)
import config


@torch.inference_mode()
def export_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    anchors_xy: torch.Tensor,
    resize_size: Tuple[int, int],
    face_thres: float,
    iou_thres: float,
    class_thres: float,
    baby_thres: float,
    device: torch.device,
    labels_map: Dict[int, str],
    out_dir: Path,
    render_original: bool = True,
) -> None:
    """
    Export predictions from the RetinaBabyFace model for a given dataset.

    This function processes a DataLoader, performs inference on each batch, and saves:
    - Annotated images with predictions drawn on them.
    - Text files containing the predictions in a standardized format.

    Args:
        model: The RetinaBabyFace model used for inference.
        loader: DataLoader providing the dataset to process.
        anchors_xy: Precomputed anchor boxes in (x, y) format.
        resize_size: Tuple (width, height) specifying the resized image dimensions.
        face_thres: Confidence threshold for face detection.
        iou_thres: IoU threshold for non-maximum suppression.
        class_thres: Confidence threshold for class predictions.
        baby_thres: Confidence threshold for baby face detection.
        device: The device (CPU/GPU) to run inference on.
        labels_map: Dictionary mapping class IDs to human-readable labels.
        out_dir: Directory where the output images and text files will be saved.
        render_original: If True, predictions are drawn on the original image resolution.
                         If False, predictions are drawn on the resized image.

    Returns:
        None. Outputs are saved to the specified directory.
    """
    # Create output directories for images and labels
    out_imgs = out_dir / "images"
    out_lbls = out_dir / "labels"
    out_imgs.mkdir(parents=True, exist_ok=True)
    out_lbls.mkdir(parents=True, exist_ok=True)

    dataset = loader.dataset
    Wr, Hr = resize_size  # Resized image dimensions (width, height)
    model.eval()  # Set the model to evaluation mode

    global_idx = 0  # Global index to track dataset samples
    for batch in tqdm(loader, desc="Export"):
        imgs = batch["image"].to(device)  # Move images to the specified device

        # Perform inference and apply rotated NMS
        outputs = infer_with_rotated_nms(
            model_or_preds=model,
            images=imgs,
            anchors_xy=anchors_xy.to(device),
            image_size=resize_size,
            face_thres=face_thres,
            baby_thres=baby_thres,
            iou_thres=iou_thres,
            class_thres=class_thres,
        )

        B = imgs.size(0)  # Batch size
        for b in range(B):
            # Get the file name and paths for the current image
            full_fname = dataset.file_list[global_idx]
            global_idx += 1
            p = Path(full_fname)
            stem = p.stem  # Base name without extension
            ext = (
                p.suffix if p.suffix else ".jpg"
            )  # Use original extension or default to .jpg

            # Determine the base image and scaling factors
            if render_original:
                # Load the original image and calculate scaling factors
                try:
                    orig_img_np, (sx, sy) = load_original_and_scale(
                        dataset, str(p), resize_size
                    )
                except Exception:
                    orig_img_np, (sx, sy) = None, (1.0, 1.0)
                if orig_img_np is None:
                    # Fallback: Open the image directly from disk
                    with Image.open(p) as im:
                        im = im.convert("RGB")
                        W0, H0 = im.size
                        sx, sy = float(W0) / float(Wr), float(H0) / float(Hr)
                        base_img = np.asarray(im)
                else:
                    base_img = orig_img_np
            else:
                # Use the resized image and no scaling
                base_img = denormalize_image(
                    imgs[b].cpu(), mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD
                )
                sx, sy = 1.0, 1.0

            # Extract predictions for the current image
            out_b = outputs[b]
            boxes_np = to_numpy(out_b["boxes"])  # (N, 5) -> θ(rad) in [:, 4]
            labels_np = to_numpy(out_b["labels"])  # (N,)
            scores_np = to_numpy(out_b["scores"])  # (N,)
            polys_np = to_numpy(
                out_b["polygons"]
            )  # (N, 8) or (N, 4, 2) in resized coords

            # Ensure polygons are in the correct (N, 4, 2) format
            polys_42 = ensure_polygons_42_shape(polys_np)

            # Reconstruct polygons if missing
            if (
                (polys_42 is None or polys_42.size == 0)
                and boxes_np is not None
                and boxes_np.size > 0
            ):
                N = boxes_np.shape[0]
                polys_42 = np.zeros((N, 4, 2), dtype=np.float32)
                for i in range(N):
                    cx, cy, w, h, th = boxes_np[i].tolist()
                    polys_42[i] = xywhr_to_poly42_shape(cx, cy, w, h, th)

            # Scale polygons to the original image resolution if needed
            if (
                render_original
                and polys_42 is not None
                and polys_42.size > 0
                and (sx != 1.0 or sy != 1.0)
            ):
                polys_for_image = polys_42.copy()
                polys_for_image[:, :, 0] *= sx
                polys_for_image[:, :, 1] *= sy
            else:
                polys_for_image = polys_42

            # Draw predictions on the image
            if polys_for_image is not None and polys_for_image.size > 0:
                angles = (
                    boxes_np[:, 4]
                    if boxes_np is not None and boxes_np.size > 0
                    else np.zeros((0,), dtype=np.float32)
                )
                painted = draw_predictions_on_image(
                    base_img=base_img,
                    polygons_xy=polys_for_image,
                    labels=labels_np,
                    scores=scores_np,
                    angles_rad=angles,
                    labels_map=labels_map,
                )
            else:
                painted = base_img

            # Save the annotated image
            Image.fromarray(painted).save(out_imgs / f"{stem}{ext}")

            # Save the predictions to a text file
            write_predictions_txt(
                out_labels_dir=out_lbls,
                stem=stem,
                boxes_xywhr=boxes_np,
                polygons_42=polys_for_image,  # Polygons in the same coordinate system as the image
                labels=labels_np,
                scores=scores_np,
            )

        # Free memory after processing the batch
        del imgs, outputs
        torch.cuda.empty_cache()

    print(f"[INFO] Export complete.\n  Images: {out_imgs}\n  Labels: {out_lbls}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export RetinaBabyFace predictions (images + txt)."
    )
    parser.add_argument("--root_dir", type=str, required=True, help="Dataset root.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="inference_export",
        help="Folder to save images and txt.",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="densenet121",
        choices=["mobilenetv1", "resnet50", "vgg16", "densenet121", "vit", "vggface2"],
    )
    parser.add_argument("--out_channel", type=int, default=config.DEFAULT_OUT_CHANNELS)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=config.DEFAULT_BATCH_SIZE)
    parser.add_argument("--face_thres", type=float, default=config.FACE_THRESH)
    parser.add_argument("--iou_thres", type=float, default=config.IOU_THRESH)
    parser.add_argument("--class_thres", type=float, default=config.CLASS_THRESH)
    parser.add_argument("--baby_thres", type=float, default=config.BABY_THRESH)
    parser.add_argument(
        "--render_original",
        action="store_true",
        help="Dibujar sobre la resolución original (recomendado).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    set_seed(42)
    device = get_default_device()
    print(f"[INFO] Device: {device}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Results will be saved to: {out_dir}")

    # Dataset/Loader
    resize_size = list(config.PRECOMPUTED_OBB_STATS.keys())[0]
    val_transform = config.get_val_transform(img_size=resize_size)
    test_dataset = BabyFacesDataset(
        root_dir=args.root_dir, split=args.split, transform=val_transform
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=custom_collate,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=seed_worker,
    )
    print(f"[INFO] Loaded {len(test_dataset)} samples from '{args.split}'.")

    # Modelo
    model = RetinaBabyFace(
        backbone_name=args.backbone, out_channel=args.out_channel, pretrained=False
    ).to(device)
    print(f"[INFO] Loading checkpoint: {args.checkpoint}")
    raw = torch.load(args.checkpoint, map_location=device)
    state = raw.get("model_state_dict", raw)
    if any(k.startswith("_orig_mod.") for k in state):
        state = {
            (k[len("_orig_mod.") :] if k.startswith("_orig_mod.") else k): v
            for k, v in state.items()
        }
    model.load_state_dict(state)
    model.eval()

    # labels
    labels_map = {
        0: "Leftside",
        1: "3/4 Leftside",
        2: "Frontal",
        3: "3/4 Rightside",
        4: "Rightside",
    }

    anchors_cache_path = config.ANCHORS_CACHE_PATH
    anchors_xy = torch.load(anchors_cache_path, map_location="cpu")[
        "anchors_xy"
    ]  # o tu forma de cargar
    print(f"[INFO] Loaded {anchors_xy.size(0)} anchors from: {anchors_cache_path}")

    # Export

    export_predictions(
        model=model,
        loader=test_loader,
        anchors_xy=anchors_xy,
        resize_size=resize_size,
        face_thres=args.face_thres,
        iou_thres=args.iou_thres,
        class_thres=args.class_thres,
        baby_thres=args.baby_thres,
        device=device,
        labels_map=labels_map,
        out_dir=out_dir,
        render_original=args.render_original,
    )

    print(f"[INFO] Listo. Archivos en: {out_dir}")


if __name__ == "__main__":
    main()
