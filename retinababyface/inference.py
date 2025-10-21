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
from engine.inference import export_predictions
from utils.helpers import (
    get_default_device,
    set_seed,
)

import config


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
        help="Backbone architecture to use",
    )
    parser.add_argument(
        "--out_channel",
        type=int,
        default=config.DEFAULT_OUT_CHANNELS,
        help="Number of output channels for FPN",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pt file).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=config.DEFAULT_BATCH_SIZE,
        help="Batch size for inference (default: 32).",
    )
    parser.add_argument(
        "--face_thres",
        type=float,
        default=config.FACE_THRESH,
        help="Confidence threshold for detections (default: 0.5).",
    )
    parser.add_argument(
        "--iou_thres",
        type=float,
        default=config.IOU_THRESH,
        help="IoU threshold for matching (default: 0.3).",
    )
    parser.add_argument(
        "--class_thres",
        type=float,
        default=config.CLASS_THRESH,
        help="Classification confidence threshold (default: 0.5).",
    )
    parser.add_argument(
        "--baby_thres",
        type=float,
        default=config.BABY_THRESH,
        help=f"Baby face confidence threshold for inference (default: {config.BABY_THRESH}).",
    )
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
