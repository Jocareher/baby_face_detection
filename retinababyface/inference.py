#!/usr/bin/env python3
import os
import argparse
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from data_setup.dataset import BabyFacesDataset
from data_setup.collate import custom_collate
from models.retinababyface import RetinaBabyFace, reset_heads
from utils.helpers import get_default_device
import config
from engine.inference import inference


def parse_args():
    """
    Parses command-line arguments for running inference with RetinaBabyFace.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run inference with RetinaBabyFace on a test set."
    )

    parser.add_argument(
        "--root_dir", type=str, required=True, help="Path to dataset root directory."
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="densenet121",
        choices=["mobilenetv1", "resnet50", "vgg16", "densenet121", "vit"],
        help="Backbone architecture to use",
    )

    parser.add_argument(
        "--out_channel", type=int, default=64, help="Number of output channels for FPN"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to use (default: test).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the model checkpoint (.pt file).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for inference (default: 32).",
    )
    parser.add_argument(
        "--conf_thres",
        type=float,
        default=0.25,
        help="Confidence threshold for detections (default: 0.25).",
    )
    parser.add_argument(
        "--iou_thres",
        type=float,
        default=0.5,
        help="IoU threshold for matching (default: 0.5).",
    )
    parser.add_argument(
        "--grid_rows",
        type=int,
        default=3,
        help="Number of rows in qualitative grid (default: 3).",
    )
    parser.add_argument(
        "--grid_cols",
        type=int,
        default=3,
        help="Number of columns in qualitative grid (default: 3).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="inference_results",
        help="Directory to save the output figures (default: inference_results/).",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    device = get_default_device()
    print(f"[INFO] Using device: {device}")

    # 1. Load test dataset and dataloader
    resize_size = list(config.PRECOMPUTED_OBB_STATS.keys())[0]
    val_transform = config.get_val_transform(img_size=resize_size)

    test_dataset = BabyFacesDataset(
        root_dir=args.root_dir,
        split=args.split,
        transform=val_transform,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=custom_collate,
        num_workers=4,
        pin_memory=True,
    )

    print(f"[INFO] Loaded {len(test_dataset)} samples from '{args.split}' split.")

    # 2. Initialize model and load checkpoint
    model = RetinaBabyFace(
        backbone_name=args.backbone,
        out_channel=args.out_channel,
        pretrained=False,
    ).to(device)

    reset_heads(model)  # reset classification/regression heads

    print(f"[INFO] Loading weights from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    model.eval()

    # 3. Define label names (without background class)
    labels_map = {
        0: "3/4 Leftside",
        1: "3/4 Rightside",
        2: "Frontal",
        3: "Left Profile",
        4: "Right Profile",
    }

    # 4. Run inference
    print("[INFO] Running inference...")
    figures = inference(
        model=model,
        checkpoint_path=args.checkpoint,  # used internally for compatibility
        test_loader=test_loader,
        device=device,
        labels_map=labels_map,
        scale_factors=config.SCALE_FACTORS,
        ratio_factors=config.RATIO_FACTORS,
        obb_stats_by_size=config.PRECOMPUTED_OBB_STATS,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        grid_shape=(args.grid_rows, args.grid_cols),
    )

    # 5. Save output figures
    os.makedirs(args.out_dir, exist_ok=True)

    figures["pr_figure"].savefig(
        os.path.join(args.out_dir, "precision_recall.png"), dpi=150
    )
    figures["confusion_figure"].savefig(
        os.path.join(args.out_dir, "confusion_matrix.png"), dpi=150
    )
    figures["grid_figure"].savefig(
        os.path.join(args.out_dir, "grid_examples.png"), dpi=150
    )

    print(f"[INFO] Inference results saved to '{args.out_dir}/'")


if __name__ == "__main__":
    main()
