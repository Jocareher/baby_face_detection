#!/usr/bin/env python3
import os
import argparse
from pathlib import Path

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
        "--output_dir",
        type=str,
        default="inferece_results",
        help="Directory to save inference results.",
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
        help="Class confidence threshold for matching (default: 0.6).",
    )
    parser.add_argument(
        "--alpha_score",
        type=float,
        default=config.ALPHA_SCORE,
        help=f"Weighting factor for combining face and orientation confidence scores (default: {config.ALPHA_SCORE}).",
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
        "--inference_results",
        type=str,
        default="inference_results",
        help="Directory to save the output figures (default: inference_results/).",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    device = get_default_device()
    print(f"[INFO] Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    figures_dir = output_dir / "figures"
    predictions_dir = output_dir / "predictions"

    figures_dir.mkdir(exist_ok=True)
    predictions_dir.mkdir(exist_ok=True)

    # 1. Load test dataset and dataloader
    resize_size = list(config.PRECOMPUTED_OBB_STATS.keys())[0]
    val_transform = config.get_val_transform(img_size=resize_size)

    norm_mean = config.IMAGENET_MEAN
    norm_std = config.IMAGENET_STD

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

    print(f"[INFO] Loading weights from: {args.checkpoint}")
    # Load the saved model weights from the checkpoint file
    raw_weights = torch.load(args.checkpoint, map_location=device)

    # Extract the state dictionary from the checkpoint
    state_dict = raw_weights.get("model_state_dict", raw_weights)

    # Check if the state dictionary contains keys prefixed with "_orig_mod."
    # This happens when the model was wrapped with torch.compile during training
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        stripped = {}
        # Remove the "_orig_mod." prefix from the keys
        for k, v in state_dict.items():
            if k.startswith("_orig_mod."):
                stripped[k[len("_orig_mod.") :]] = v
            else:
                stripped[k] = v
        state_dict = stripped

    # Load the state dictionary into the model
    model.load_state_dict(state_dict)
    model.eval()

    # 3. Define label names (without background class)
    labels_map = {
        0: "Leftside",
        1: "3/4 Leftside",
        2: "Frontal",
        3: "3/4 Rightside",
        4: "Rightside",
    }
    # 4. Run inference
    print("[INFO] Running inference...")
    figures = inference(
        model=model,
        test_loader=test_loader,
        output_dir=predictions_dir,
        device=device,
        labels_map=labels_map,
        scale_factors=config.SCALE_FACTORS,
        ratio_factors=config.RATIO_FACTORS,
        face_thres=args.face_thres,
        iou_thres=args.iou_thres,
        class_thres=args.class_thres,
        grid_shape=(args.grid_rows, args.grid_cols),
        mean=norm_mean,
        std=norm_std,
    )

    # 5. Save output figures
    # Save all figures
    figures["pr_figure"].savefig(figures_dir / "precision_recall.png", dpi=150)
    figures["confusion_figure_raw"].savefig(
        figures_dir / "confusion_matrix_raw.png", dpi=150
    )
    figures["confusion_figure_normalized"].savefig(
        figures_dir / "confusion_matrix_normalized.png", dpi=150
    )
    figures["grid_figure"].savefig(figures_dir / "grid_examples.png", dpi=150)
    figures["iou_boxplot_figure"].savefig(figures_dir / "iou_boxplot.png", dpi=150)
    figures["angle_boxplot_figure"].savefig(figures_dir / "angle_boxplot.png", dpi=150)
    figures["f1_threshold_figure"].savefig(figures_dir / "f1_threshold.png", dpi=150)

    print(f"[INFO] All figures saved to {figures_dir}")
    print(f"[INFO] All predictions saved to {predictions_dir}")


if __name__ == "__main__":
    main()
