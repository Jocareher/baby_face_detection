import argparse
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as T

from data_setup.dataset import BabyFacesDataset, ImageFolderDataset
from data_setup.collate import custom_collate, images_only_collate
from models.retinababyface import RetinaBabyFace
from engine.inference import (
    infer_with_rotated_nms,
    load_original_and_scale,
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
)
import config


@torch.inference_mode()
def export_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    anchors_xy: torch.Tensor,
    resize_size: Tuple[int, int],     # (W, H) usado por el modelo
    face_thres: float,
    iou_thres: float,
    class_thres: float,
    baby_thres: float,
    device: torch.device,
    labels_map: Dict[int, str],
    out_dir: Path,
    output_scale: str = "original",   # "original" | "resized"
) -> None:
    """
    Infiere y guarda imágenes + txt en la MISMA escala elegida:
      - "original": resolución nativa del archivo en disco.
      - "resized" : la resolución de entrada del modelo (p.ej., 640x640).

    Funciona con:
      - Tu BabyFacesDataset + custom_collate (sin 'meta' requerido)
      - Dataset de solo imágenes + images_only_collate (sin 'meta', sin 'paths' requerido)
    """

    out_imgs = Path(out_dir) / "images"
    out_lbls = Path(out_dir) / "labels"
    out_imgs.mkdir(parents=True, exist_ok=True)
    out_lbls.mkdir(parents=True, exist_ok=True)

    model.eval()
    anchors_xy = anchors_xy.to(device, non_blocking=True)
    Wr, Hr = resize_size
    nms_image_size = (Hr, Wr)  # la mayoría del post-proceso espera (H, W)

    dataset = loader.dataset
    processed = saved = empty_batches = no_dets = errors = 0
    global_idx = 0
    warned_fallback_resized = False  # avisar una sola vez si pediste "original" y no hay archivo

    tqdm.write(f"🧠  Device: {device}")
    tqdm.write(f"📦  Dataloader: {len(loader)} batches | bs={getattr(loader,'batch_size','?')}")
    tqdm.write(f"📐  Resize (W,H)={resize_size} | NMS(H,W)={nms_image_size}")
    tqdm.write(f"🗃️  Output: {out_imgs} / {out_lbls}")
    tqdm.write(f"📏  Output scale: {output_scale}")

    for batch in tqdm(loader, desc="⚙️  Batches", unit="batch"):
        imgs = batch["image"].to(device, non_blocking=True)
        if imgs.numel() == 0:
            empty_batches += 1
            continue

        # Inference + postproc
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
            tqdm.write(f"❌  Inference error: {e}")
            global_idx += imgs.size(0)
            continue

        B = imgs.size(0)
        for b in range(B):
            processed += 1

            # 1) Resolver path ABSOLUTO usando *tu* resolve_image_path (sin reconstrucciones)
            p = resolve_image_path(batch, b, global_idx, dataset=dataset)  # <- TU función
            # p puede ser un fallback tipo sample_XXXX.jpg (no existente en disco)

            # 2) Base image + escala
            try:
                if output_scale == "original" and p.is_file():
                    with Image.open(p) as im:
                        im = im.convert("RGB")
                        W0, H0 = im.size
                        base_img = np.asarray(im)   # original
                    sx, sy = float(W0) / float(Wr), float(H0) / float(Hr)
                    stem, ext = p.stem, (p.suffix if p.suffix else ".jpg")
                else:
                    # 'resized' o pediste 'original' pero p no existe → caemos a resized y avisamos una vez
                    if output_scale == "original" and not p.is_file() and not warned_fallback_resized:
                        tqdm.write("⚠️  Requested 'original' scale but no absolute file on disk; drawing/saving in 'resized' scale.")
                        warned_fallback_resized = True

                    base_img = denormalize_image(imgs[b])  # tu función
                    sx, sy = 1.0, 1.0
                    stem, ext = (p.stem, (p.suffix if p.suffix else ".jpg")) if p is not None else (f"sample_{processed:06d}", ".jpg")
            except Exception as e:
                errors += 1
                tqdm.write(f"❌  Base image error ({p}): {e}")
                global_idx += 1
                continue

            # 3) Predicciones de esta imagen
            out_b = outputs[b]
            boxes_np  = to_numpy(out_b.get("boxes"))     # (N,5) -> cx,cy,w,h,theta
            labels_np = to_numpy(out_b.get("labels"))
            scores_np = to_numpy(out_b.get("scores"))
            polys_np  = to_numpy(out_b.get("polygons"))  # (N,8) o (N,4,2)

            polys_42 = ensure_polygons_42_shape(polys_np)
            if (polys_42 is None or polys_42.size == 0) and boxes_np is not None and boxes_np.size > 0:
                N = boxes_np.shape[0]
                polys_42 = np.stack([xywhr_to_poly42_shape(*boxes_np[i]) for i in range(N)], axis=0).astype(np.float32)

            # 4) Escala de guardado
            if output_scale == "original" and (sx != 1.0 or sy != 1.0):
                polys_for_img = scale_polys(polys_42, sx, sy)
                boxes_for_txt = scale_xywhr_boxes(boxes_np, sx, sy) if boxes_np is not None else None
            else:
                polys_for_img = polys_42
                boxes_for_txt = boxes_np

            # 5) Dibujar y guardar
            try:
                if polys_for_img is not None and polys_for_img.size > 0:
                    angles = boxes_np[:, 4] if (boxes_np is not None and boxes_np.size > 0) else np.zeros((0,), dtype=np.float32)
                    lbls = labels_np if labels_np is not None else np.zeros((polys_for_img.shape[0],), dtype=np.int64)
                    scrs = scores_np if scores_np is not None else np.zeros((polys_for_img.shape[0],), dtype=np.float32)
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
                tqdm.write(f"❌  Save error ({stem}{ext}): {e}")

            global_idx += 1

    # resumen
    tqdm.write("✅  Export complete")
    tqdm.write(f"   • Processed : {processed}")
    tqdm.write(f"   • Saved     : {saved}")
    tqdm.write(f"   • No dets   : {no_dets}")
    tqdm.write(f"   • Empty bch : {empty_batches}")
    tqdm.write(f"   • Errors    : {errors}")
    tqdm.write(f"📂 Images: {out_imgs}")
    tqdm.write(f"📝 Labels: {out_lbls}")

def parse_args():
    parser = argparse.ArgumentParser(description="Export RetinaBabyFace predictions (images + txt).")
    parser.add_argument("--root_dir",   type=str, help="Dataset root (for BabyFacesDataset mode).")
    parser.add_argument("--images_dir", type=str, help="Images-only mode: directory with images (recursively).")
    parser.add_argument("--output_dir", type=str, default="inference_export", help="Folder to save images and txt.")
    parser.add_argument("--backbone",   type=str, default="densenet121",
                        choices=["mobilenetv1", "resnet50", "vgg16", "densenet121", "vit", "vggface2"])
    parser.add_argument("--out_channel", type=int, default=128)  # replace with config.DEFAULT_OUT_CHANNELS if you have it
    parser.add_argument("--split", type=str, default="test", choices={"train", "val", "test"},
                        help="Split to load from BabyFacesDataset.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=config.DEFAULT_BATCH_SIZE)
    parser.add_argument("--face_thres", type=float, default=config.FACE_THRESH)
    parser.add_argument("--iou_thres",  type=float, default=config.IOU_THRESH)
    parser.add_argument("--class_thres", type=float, default=config.CLASS_THRESH)
    parser.add_argument("--baby_thres",  type=float, default=config.BABY_THRESH)
    parser.add_argument("--output_scale", type=str, default="original", choices={"original", "resized"},
                        help="Save images and TXT in 'original' image coords or in resized coords (e.g., 640x640).")
    args = parser.parse_args()

    # Enforce exactly one mode
    if (args.split is None) == (args.images_dir is None):
        parser.error("You must provide exactly one of: --split OR --images-dir")
    if args.split and not args.root_dir:
        parser.error("--root-dir is required when using --split")
    return args

def main():
    args = parse_args()

    set_seed(42)
    device = get_default_device()
    print(f"[INFO] Device: {device}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Results will be saved to: {out_dir}")

    # === Choose resize_size (W,H) consistent with your training ===
    resize_size = list(config.PRECOMPUTED_OBB_STATS.keys())[0] # replace with your config.PRECOMPUTED_OBB_STATS key if you have it

    # === Build dataset/loader in the selected mode ===
    if args.images_dir is not None:
        # Images-only (no labels)
        safe_transform = T.Compose([T.Resize(resize_size), T.ToTensor(), T.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)])
        dataset = ImageFolderDataset(images_dir=args.images_dir, transform=safe_transform)
        loader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            collate_fn=images_only_collate,
                            num_workers=4,
                            pin_memory=True)
        print(f"[INFO] Loaded {len(dataset)} images from '{args.images_dir}' (images-only mode).")
    else:
        # BabyFacesDataset mode (uses your dataset + collate + transform)
        val_transform = config.get_val_transform(img_size=resize_size)  # <-- your validated transform
        dataset = BabyFacesDataset(root_dir=args.root_dir, split=args.split, transform=val_transform)
        loader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            collate_fn=custom_collate,
                            num_workers=4,
                            pin_memory=True)
        print(f"[INFO] Loaded {len(dataset)} samples from split '{args.split}'.")

    # === Model ===place
    model = RetinaBabyFace(backbone_name=args.backbone, out_channel=args.out_channel, pretrained=False).to(device)
    print(f"[INFO] Loading checkpoint: {args.checkpoint}")
    raw = torch.load(args.checkpoint, map_location=device)
    state = raw.get("model_state_dict", raw)
    if any(k.startswith("_orig_mod.") for k in state):
        state = {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()

    labels_map = {0: "Leftside", 1: "3/4 Leftside", 2: "Frontal", 3: "3/4 Rightside", 4: "Rightside"}

    # === Anchors ===
    # Replace this with your own path/load method.
    anchors_cache_path = "weights/anchors_cache.pt"  # <-- replace
    anchors_xy = torch.load(anchors_cache_path, map_location="cpu")["anchors_xy"]
    print(f"[INFO] Loaded {anchors_xy.size(0)} anchors from: {anchors_cache_path}")

    # === Export ===
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
        output_scale=args.output_scale,  # "original" or "resized"
    )

    print(f"[INFO] Done. Files in: {out_dir}")

if __name__ == "__main__":
    main()
