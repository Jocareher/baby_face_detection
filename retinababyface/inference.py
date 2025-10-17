import argparse
from pathlib import Path
import math
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as T

from data_setup.dataset import ImageFolderDataset
from data_setup.collate import images_only_collate
from models.retinababyface import RetinaBabyFace
from engine.inference import (
    infer_with_rotated_nms,
    denormalize_image,
)
from utils.helpers import (
    get_default_device,
    set_seed,
    to_numpy,
    ensure_polygons_42_shape,
    resolve_image_path,
)
from utils.visualize import (
    draw_predictions_on_image,
    write_predictions_txt,
    xywhr_to_poly42_shape,
    scale_polys,
    scale_xywhr_boxes,
    crop_obb,
    polygon_to_size
)
import config


@torch.inference_mode()
def export_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    anchors_xy: torch.Tensor,
    resize_size: Tuple[int, int],  # (W, H) used by model
    face_thres: float,
    iou_thres: float,
    class_thres: float,
    baby_thres: float,
    device: torch.device,
    labels_map: Dict[int, str],
    out_dir: Path,
    output_scale: str = "original",  # "original" | "resized"
) -> None:
    """
    Export model predictions as annotated images and text files.

    This function performs inference on a dataset and saves:
    1. Annotated images with detected faces and their orientations
    2. Text files with detection coordinates and metadata

    The output can be saved in two coordinate scales:
    - "original": Native resolution of input images
    - "resized": Model input resolution (e.g., 640x640)

    Args:
        model: The RetinaBabyFace detection model
        loader: DataLoader providing images (and optionally labels)
        anchors_xy: Anchor box coordinates tensor
        resize_size: Model input size as (width, height)
        face_thres: Face detection confidence threshold
        iou_thres: IoU threshold for NMS
        class_thres: Classification confidence threshold
        baby_thres: Baby face confidence threshold
        device: Torch device to run inference on
        labels_map: Dictionary mapping class indices to names
        out_dir: Base output directory path
        output_scale: Whether to save in "original" or "resized" coordinates

    Outputs:
        - out_dir/images/: Directory with annotated images
        - out_dir/labels/: Directory with text files containing detections
    """

    # Create output directories
    out_imgs = Path(out_dir) / "images"
    out_lbls = Path(out_dir) / "labels"
    out_imgs.mkdir(parents=True, exist_ok=True)
    out_lbls.mkdir(parents=True, exist_ok=True)
    out_crops = Path(out_dir) / "crops"
    out_imgs.mkdir(parents=True, exist_ok=True)
    out_lbls.mkdir(parents=True, exist_ok=True)
    out_crops.mkdir(parents=True, exist_ok=True)

    # Setup model and anchors
    model.eval()
    anchors_xy = anchors_xy.to(device, non_blocking=True)
    Wr, Hr = resize_size
    nms_image_size = (Hr, Wr)  # NMS expects (H,W) format

    dataset = loader.dataset

    # Statistics counters
    processed = 0  # Total images processed
    saved = 0  # Successfully saved pairs
    empty_batches = 0  # Batches with no valid images
    no_dets = 0  # Images with no detections
    errors = 0  # Failed operations

    # Print configuration
    tqdm.write(f"🧠  Inference on device: {device}")
    tqdm.write(
        f"📦  Dataloader: {len(loader)} batches | batch_size={getattr(loader, 'batch_size', '?')}"
    )
    tqdm.write(f"🗋  Output dir: {out_dir}  →  images/, labels/")
    tqdm.write(f"📐  Resize size (W,H): {resize_size} | NMS uses (H,W)={nms_image_size}")
    tqdm.write(f"📏  Output scale: {output_scale}")

    with tqdm(total=len(loader), desc="⚙️  Batches", unit="batch") as pbar_batches:
        global_idx = 0

        for batch in loader:
            # Move images to device
            imgs = batch["image"].to(device, non_blocking=True)
            if imgs.numel() == 0:
                empty_batches += 1
                pbar_batches.update(1)
                continue

            # Run inference and NMS
            try:
                outputs = infer_with_rotated_nms(
                    model_or_preds=model,
                    images=imgs,
                    anchors_xy=anchors_xy,
                    image_size=nms_image_size,
                    face_thres=face_thres,
                    baby_thres=baby_thres,
                    iou_thres=iou_thres,
                    class_thres=class_thres,
                )
            except Exception as e:
                errors += 1
                tqdm.write(f"❌  Inference error in batch: {e}")
                pbar_batches.update(1)
                continue

            # Process each image in batch
            B = imgs.size(0)
            with tqdm(
                total=B, desc="   🖼️  Images", leave=False, unit="img"
            ) as pbar_imgs:
                for b in range(B):
                    processed += 1

                    # Get robust path for image
                    p = resolve_image_path(batch, b, global_idx, dataset=dataset)
                    stem, ext = p.stem, (p.suffix if p.suffix else ".jpg")

                    # Load base image and compute scale factors
                    try:
                        if output_scale == "original" and p.exists():
                            # Load original image
                            with Image.open(p) as im:
                                im = im.convert("RGB")
                                W0, H0 = im.size
                                base_img = np.asarray(im)
                            sx, sy = float(W0) / float(Wr), float(H0) / float(Hr)
                        else:
                            # Use resized tensor or fallback
                            base_img = denormalize_image(imgs[b])
                            sx, sy = 1.0, 1.0
                            if not ext:
                                ext = ".jpg"
                    except Exception as e:
                        errors += 1
                        tqdm.write(f"❌  Could not prepare base image for {p}: {e}")
                        pbar_imgs.update(1)
                        global_idx += 1
                        continue

                    # Extract predictions
                    try:
                        out_b = outputs[b]
                        boxes_np = to_numpy(
                            out_b.get("boxes")
                        )  # (N,5) -> cx,cy,w,h,theta
                        labels_np = to_numpy(out_b.get("labels"))
                        scores_np = to_numpy(out_b.get("scores"))
                        polys_np = to_numpy(out_b.get("polygons"))  # (N,8) or (N,4,2)

                        # Normalize/reconstruct polygons if needed
                        polys_42 = ensure_polygons_42_shape(polys_np)
                        if (
                            (polys_42 is None or polys_42.size == 0)
                            and boxes_np is not None
                            and boxes_np.size > 0
                        ):
                            N = boxes_np.shape[0]
                            polys_42 = np.stack(
                                [xywhr_to_poly42_shape(*boxes_np[i]) for i in range(N)],
                                axis=0,
                                dtype=np.float32,
                            )
                    except Exception as e:
                        errors += 1
                        tqdm.write(f"❌  Postprocess error for {p}: {e}")
                        pbar_imgs.update(1)
                        global_idx += 1
                        continue

                    # Scale coordinates to target output size
                    if output_scale == "original" and (sx != 1.0 or sy != 1.0):
                        polys_for_img = scale_polys(polys_42, sx, sy)
                        boxes_for_txt = (
                            scale_xywhr_boxes(boxes_np, sx, sy)
                            if boxes_np is not None
                            else None
                        )
                    else:
                        polys_for_img = polys_42
                        boxes_for_txt = boxes_np

                    # Save results
                    try:
                        if polys_for_img is not None and polys_for_img.size > 0:
                            angles = (
                                boxes_np[:, 4]
                                if (boxes_np is not None and boxes_np.size > 0)
                                else np.zeros((0,), dtype=np.float32)
                            )
                            lbls = (
                                labels_np
                                if labels_np is not None
                                else np.zeros((polys_for_img.shape[0],), dtype=np.int64)
                            )
                            scrs = (
                                scores_np
                                if scores_np is not None
                                else np.zeros(
                                    (polys_for_img.shape[0],), dtype=np.float32
                                )
                            )
                            painted = draw_predictions_on_image(
                                base_img=base_img,
                                polygons_xy=polys_for_img,
                                labels=lbls,
                                scores=scrs,
                                angles_rad=angles,
                                labels_map=labels_map,
                            )
                        else:
                            painted = base_img
                            no_dets += 1

                        Image.fromarray(painted).save(out_imgs / f"{stem}{ext}")

                        write_predictions_txt(
                            out_labels_dir=out_lbls,
                            stem=stem,
                            boxes_xywhr=boxes_for_txt,
                            polygons_42=polys_for_img,
                            labels=labels_np,
                            scores=scores_np,
                        )
                        saved += 1

                    except Exception as e:
                        errors += 1
                        tqdm.write(f"❌  Saving error for {p}: {e}")
                    
                    try:
                        if polys_for_img is not None and polys_for_img.size > 0:
                            N = polys_for_img.shape[0]
                            pil_base = Image.fromarray(base_img)  # para transform QUAD

                            for j in range(N):
                                poly = polys_for_img[j].astype(np.float32)  # (4,2) en el MISMO orden predicho
                                # longitudes de lados adyacentes según el orden original v0->v1, v1->v2
                                len01 = float(np.linalg.norm(poly[1] - poly[0]))
                                len12 = float(np.linalg.norm(poly[2] - poly[1]))
                                # margen opcional para que haya contexto tras rotar (ajusta si quieres)
                                scale_crop = 1.20
                                wj = max(2, int(round(scale_crop * len01)))
                                hj = max(2, int(round(scale_crop * len12)))

                                # QUAD de PIL mapea destino (0,0)-(w,0)-(w,h)-(0,h) a esta CUÁDRUPLA fuente
                                quad = poly.reshape(-1).tolist()  # (x0,y0,x1,y1,x2,y2,x3,y3) en el orden original
                                crop = pil_base.transform(
                                    (wj, hj),
                                    Image.QUAD,
                                    data=quad,
                                    resample=Image.BILINEAR,
                                )

                                # Ángulo (rad → deg). Signo: + antihorario, − horario (PIL ya sigue esto)
                                if boxes_for_txt is not None and boxes_for_txt.size > 0:
                                    theta_deg = math.degrees(float(boxes_for_txt[j, 4]))
                                else:
                                    theta_deg = 0.0

                                # Rotar alrededor del centro; expand=True para no recortar contenido
                                crop_rot = crop.rotate(theta_deg, expand=True, resample=Image.BILINEAR)

                                # ---- Resize "COVER" + center-crop a 640x640 (Wr,Hr) ----
                                # Queremos llenar por completo 640x640 sin letterbox negro.
                                Wr, Hr = resize_size  # p.ej., (640, 640)
                                rw, rh = crop_rot.size
                                ratio = max(Wr / float(rw), Hr / float(rh))  # COVER
                                new_w = int(math.ceil(rw * ratio))
                                new_h = int(math.ceil(rh * ratio))
                                cover = crop_rot.resize((new_w, new_h), resample=Image.BILINEAR)
                                left = max(0, (new_w - Wr) // 2)
                                top  = max(0, (new_h - Hr) // 2)
                                crop_final = cover.crop((left, top, left + Wr, top + Hr))  # 640x640 lleno

                                # Carpeta por class_idx (o cambia a labels_map si prefieres nombre)
                                cls = int(labels_np[j]) if (labels_np is not None and labels_np.size > j) else 0
                                cls_dir = (Path(out_dir) / "crops" / f"{cls}")
                                cls_dir.mkdir(parents=True, exist_ok=True)

                                crop_name = f"{stem}_{j:02d}.jpg"
                                crop_final.save(cls_dir / crop_name)
                    except Exception as e:
                        errors += 1
                        tqdm.write(f"❌  Crop error ({stem}): {e}")
                        

                    pbar_imgs.update(1)
                    global_idx += 1

            pbar_batches.update(1)

    # Print summary statistics
    tqdm.write("✅  Export complete")
    tqdm.write(f"   • Processed images : {processed}")
    tqdm.write(f"   • Saved (img+txt)  : {saved}")
    tqdm.write(f"   • No detections    : {no_dets}")
    tqdm.write(f"   • Empty batches    : {empty_batches}")
    tqdm.write(f"   • Errors           : {errors}")
    tqdm.write(f"📂  Images: {out_imgs}")
    tqdm.write(f"📝  Labels: {out_lbls}")
    tqdm.write(f"✂️  Crops : {out_crops}/<class_idx>/")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export RetinaBabyFace predictions (images + txt)."
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        help="Images-only mode: directory with images (recursively).",
    )
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
    parser.add_argument(
        "--out_channel",
        type=int,
        default=config.DEFAULT_OUT_CHANNELS,
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=config.DEFAULT_BATCH_SIZE)
    parser.add_argument("--face_thres", type=float, default=config.FACE_THRESH)
    parser.add_argument("--iou_thres", type=float, default=config.IOU_THRESH)
    parser.add_argument("--class_thres", type=float, default=config.CLASS_THRESH)
    parser.add_argument("--baby_thres", type=float, default=config.BABY_THRESH)
    parser.add_argument(
        "--output_scale",
        type=str,
        default="original",
        choices={"original", "resized"},
        help="Save images and TXT in 'original' image coords or in resized coords (e.g., 640x640).",
    )
    args = parser.parse_args()

    return args


def main():
    # Parse command line arguments
    args = parse_args()

    # Set random seed and get device (CPU/GPU)
    set_seed(42)
    device = get_default_device()
    print(f"\n🖥️  Using device: {device}")

    # Create output directory
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {out_dir}")

    # === Model Input Size Configuration ===
    # Get resize dimensions (W,H) from config - should match training settings
    resize_size = list(config.PRECOMPUTED_OBB_STATS.keys())[0]
    print(f"🔄 Model input size (W,H): {resize_size}")

    # === Dataset/DataLoader Setup ===
    # Configure image transformations (resize, normalize)
    test_transform = T.Compose(
        [
            T.Resize(resize_size),
            T.ToTensor(),
            T.Normalize(mean=config.MEAN, std=config.STD),
        ]
    )

    # Create dataset and dataloader for inference
    dataset = ImageFolderDataset(images_dir=args.images_dir, transform=test_transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=images_only_collate,
        num_workers=4,
        pin_memory=True,
    )
    print(f"📊 Loaded {len(dataset)} images from: {args.images_dir}")

    # === Model Initialization ===
    # Create model and load checkpoint
    print(f"\n🔧 Initializing {args.backbone} backbone...")
    model = RetinaBabyFace(
        backbone_name=args.backbone, out_channel=args.out_channel, pretrained=False
    ).to(device)

    print(f"📥 Loading checkpoint: {args.checkpoint}")
    raw = torch.load(args.checkpoint, map_location=device)
    state = raw.get("model_state_dict", raw)

    # Handle checkpoint format variations
    if any(k.startswith("_orig_mod.") for k in state):
        state = {
            (k[len("_orig_mod.") :] if k.startswith("_orig_mod.") else k): v
            for k, v in state.items()
        }
    model.load_state_dict(state)
    model.eval()

    # Define face orientation labels
    labels_map = {
        0: "Leftside",
        1: "3/4 Leftside",
        2: "Frontal",
        3: "3/4 Rightside",
        4: "Rightside",
    }

    # === Anchor Boxes ===
    anchors_cache_path = config.ANCHORS_CACHE_PATH
    anchors_xy = torch.load(anchors_cache_path, map_location="cpu")["anchors_xy"]
    print(f"⚓ Loaded {anchors_xy.size(0)} anchor boxes")

    # === Run Inference ===
    print("\n🚀 Starting inference...")
    export_predictions(
        model=model,
        loader=loader,
        anchors_xy=anchors_xy,
        resize_size=resize_size,
        face_thres=args.face_thres,
        iou_thres=args.iou_thres,
        class_thres=args.class_thres,
        baby_thres=args.baby_thres,
        device=device,
        labels_map=labels_map,
        out_dir=out_dir,
        output_scale=args.output_scale,
    )

    print(f"\n✨ Inference complete! Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
