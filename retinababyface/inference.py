import argparse
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image, ImageDraw

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
from utils.helpers import get_default_device, seed_worker, set_seed
import config


import math
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


# ---------------------- helpers de forma ------------------------------------
def _as_np(x):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _ensure_polygons_42(polys_np: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """
    Acepta polígonos en (N,8) o (N,4,2) y devuelve (N,4,2) float32. Si está vacío, None.
    """
    if polys_np is None:
        return None
    polys_np = _as_np(polys_np)
    if polys_np.size == 0:
        return None
    if polys_np.ndim == 2 and polys_np.shape[1] == 8:
        return polys_np.reshape(-1, 4, 2).astype(np.float32)
    if polys_np.ndim == 3 and polys_np.shape[1:] == (4, 2):
        return polys_np.astype(np.float32)
    raise ValueError(f"Polygons shape no soportado: {polys_np.shape}")


def _xywhr_to_poly42(cx, cy, w, h, theta):
    dx, dy = w / 2.0, h / 2.0
    c, s = math.cos(theta), math.sin(theta)
    base = [(-dx, -dy), (dx, -dy), (dx, dy), (-dx, dy)]  # v0..v3 (frente v0->v1)
    pts = []
    for x, y in base:
        px = cx + x * c - y * s
        py = cy + x * s + y * c
        pts.append((px, py))
    return np.asarray(pts, dtype=np.float32)  # (4,2)


# ---------------------- dibujo ------------------------------------------------
def draw_predictions_on_image(
    base_img: np.ndarray,  # np.uint8 (H,W,3) en la resolución que elegiste
    polygons_xy: np.ndarray,  # (N,4,2) en coords de base_img
    labels: np.ndarray,  # (N,)
    scores: np.ndarray,  # (N,)
    angles_rad: np.ndarray,  # (N,) ángulos en radianes (AngleHead)
    labels_map: Dict[int, str],
) -> np.ndarray:
    img = Image.fromarray(base_img.copy())
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    N = polygons_xy.shape[0]
    for i in range(N):
        poly = polygons_xy[i]
        # asegurar ints
        poly_pts = [
            (int(round(float(x))), int(round(float(y)))) for x, y in poly.tolist()
        ]
        # contorno azul
        draw.line(poly_pts + [poly_pts[0]], width=2, fill=(0, 64, 160))
        # arista frontal roja (v0->v1)
        draw.line([poly_pts[0], poly_pts[1]], width=2, fill=(160, 0, 0))

        # texto en top-left del OBB
        tl_x = min(p[0] for p in poly_pts)
        tl_y = min(p[1] for p in poly_pts)
        ang_deg = math.degrees(float(angles_rad[i]))
        lbl_name = labels_map.get(int(labels[i]), str(int(labels[i])))
        txt = f"{lbl_name}: {ang_deg:.1f}° / {float(scores[i]):.2f}"
        # texto con trazo para legibilidad
        draw.text(
            (tl_x, tl_y),
            txt,
            fill=(255, 255, 255),
            font=font,
            stroke_width=2,
            stroke_fill=(0, 64, 160),
        )
    return np.asarray(img)


# ---------------------- escritura TXT ---------------------------------------
def write_predictions_txt(
    out_labels_dir: Path,
    stem: str,
    boxes_xywhr: Optional[np.ndarray],  # (N,5) -> cx,cy,w,h,theta (rad)
    polygons_42: Optional[np.ndarray],  # (N,4,2) en coords de la imagen final
    labels: np.ndarray,  # (N,)
    scores: np.ndarray,  # (N,)
) -> None:
    out_labels_dir.mkdir(parents=True, exist_ok=True)

    boxes_np = _as_np(boxes_xywhr) if boxes_xywhr is not None else None
    labels_np = (
        _as_np(labels).astype(np.int64)
        if labels is not None
        else np.zeros((0,), dtype=np.int64)
    )
    scores_np = (
        _as_np(scores).astype(np.float32)
        if scores is not None
        else np.zeros((0,), dtype=np.float32)
    )
    polys_42 = _ensure_polygons_42(polygons_42)

    # N consistente
    Ns = []
    if boxes_np is not None:
        Ns.append(boxes_np.shape[0])
    if polys_42 is not None:
        Ns.append(polys_42.shape[0])
    if labels_np is not None:
        Ns.append(labels_np.shape[0])
    if scores_np is not None:
        Ns.append(scores_np.shape[0])
    N = min(Ns) if Ns else 0

    # reconstruir polígonos si no vinieron
    if N > 0 and polys_42 is None:
        assert (
            boxes_np is not None and boxes_np.shape[1] == 5
        ), "Necesito boxes (N,5) para reconstruir polígonos."
        polys_42 = np.zeros((N, 4, 2), dtype=np.float32)
        for i in range(N):
            cx, cy, w, h, th = boxes_np[i].tolist()
            polys_42[i] = _xywhr_to_poly42(cx, cy, w, h, th)

    angles_rad = (
        boxes_np[:N, 4]
        if boxes_np is not None and boxes_np.size
        else np.zeros((N,), dtype=np.float32)
    )
    labels_np = labels_np[:N]
    scores_np = scores_np[:N]

    # escribir
    txt_path = out_labels_dir / f"{stem}.txt"
    with open(txt_path, "w") as f:
        for i in range(N):
            x1, y1 = polys_42[i, 0]
            x2, y2 = polys_42[i, 1]
            x3, y3 = polys_42[i, 2]
            x4, y4 = polys_42[i, 3]
            f.write(
                f"{int(labels_np[i])} "
                f"{int(round(x1))} {int(round(y1))} "
                f"{int(round(x2))} {int(round(y2))} "
                f"{int(round(x3))} {int(round(y3))} "
                f"{int(round(x4))} {int(round(y4))} "
                f"{float(angles_rad[i]):.6f} {float(scores_np[i]):.6f}\n"
            )


# ---------------------- export main -----------------------------------------
@torch.no_grad()
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
    Recorre el loader y, por imagen:
      - guarda la imagen pintada en out_dir/images/
      - guarda el .txt en out_dir/labels/
    Si render_original=True, pinta/guarda con coords y tamaño de la imagen original.
    Si False, pinta/guarda en 640x640 (o el resize_size que uses).
    """
    out_imgs = out_dir / "images"
    out_lbls = out_dir / "labels"
    out_imgs.mkdir(parents=True, exist_ok=True)
    out_lbls.mkdir(parents=True, exist_ok=True)

    dataset = loader.dataset
    Wr, Hr = resize_size  # (W,H)
    model.eval()

    global_idx = 0
    for batch in tqdm(loader, desc="Export"):
        imgs = batch["image"].to(device)

        # inferencia + NMS
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

        B = imgs.size(0)
        for b in range(B):
            # nombre/paths
            full_fname = dataset.file_list[
                global_idx
            ]  # suele ser ruta absoluta o relativa
            global_idx += 1
            p = Path(full_fname)
            stem = p.stem
            ext = p.suffix if p.suffix else ".jpg"

            # elegir base_img y escalas
            if render_original:
                # usa tu helper si lo tienes disponible:
                try:
                    orig_img_np, (sx, sy) = load_original_and_scale(
                        dataset, str(p), resize_size
                    )
                except Exception:
                    orig_img_np, (sx, sy) = None, (1.0, 1.0)
                if orig_img_np is None:
                    # fallback: abrir desde disco
                    with Image.open(p) as im:
                        im = im.convert("RGB")
                        W0, H0 = im.size
                        sx, sy = float(W0) / float(Wr), float(H0) / float(Hr)
                        base_img = np.asarray(im)
                else:
                    base_img = orig_img_np
            else:
                # pintar en 640x640 (o resize_size): desnormalizar
                base_img = denormalize_image(
                    imgs[b].cpu(), mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD
                )
                sx, sy = 1.0, 1.0

            # recoger salidas
            out_b = outputs[b]
            boxes_np = _as_np(out_b["boxes"])  # (N,5) -> θ(rad) en [:,4]
            labels_np = _as_np(out_b["labels"])  # (N,)
            scores_np = _as_np(out_b["scores"])  # (N,)
            polys_np = _as_np(out_b["polygons"])  # (N,8) o (N,4,2) en coords de RESIZE

            polys_42 = _ensure_polygons_42(polys_np)

            # reconstruir si falta
            if (
                (polys_42 is None or polys_42.size == 0)
                and boxes_np is not None
                and boxes_np.size > 0
            ):
                N = boxes_np.shape[0]
                polys_42 = np.zeros((N, 4, 2), dtype=np.float32)
                for i in range(N):
                    cx, cy, w, h, th = boxes_np[i].tolist()
                    polys_42[i] = _xywhr_to_poly42(cx, cy, w, h, th)

            # escalar a original si corresponde
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

            # pintar
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

            # guardar imagen (conservar extensión si la tenemos; si no, .jpg)
            Image.fromarray(painted).save(out_imgs / f"{stem}{ext}")

            # guardar TXT (usar polígonos en el MISMO sistema en el que pintaste)
            write_predictions_txt(
                out_labels_dir=out_lbls,
                stem=stem,
                boxes_xywhr=boxes_np,
                polygons_42=polys_for_image,  # ya en sistema de coords de la imagen guardada
                labels=labels_np,
                scores=scores_np,
            )

        del imgs, outputs
        torch.cuda.empty_cache()

    print(f"[INFO] Export completo.\n  Imágenes: {out_imgs}\n  Labels:   {out_lbls}")


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
