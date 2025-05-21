# This script is used to train the RetinaBabyFace model on the BabyFace dataset.
# It includes data loading, augmentation, model definition, and training loop.
# The script uses PyTorch and torchvision for model training and data handling.
# The RetinaBabyFace model is a custom architecture designed for face detection and recognition tasks.

import argparse
import os
import sys
from pathlib import Path

# Adding the root directory to the system path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
    print(f"[INFO] Adding {ROOT_DIR} to sys.path")

import yaml
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchinfo import summary

from data_setup.dataset import BabyFacesDataset
from data_setup.collate import custom_collate
from models.mobilenet import MobileNetV1
from models.retinababyface import RetinaBabyFace, reset_heads
from utils.helpers import set_seed, get_default_device
from engine.train import train, EarlyStopping
from engine.inference import inference
from loss.losses import MultiTaskLoss
from utils.visualize import visualize_and_save_dataset_in_script, create_training_gif
import config


import argparse
import config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate RetinaBabyFace model"
    )

    # Dataset
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Path to the dataset root directory (containing train/val/test subfolders).",
    )

    # Model & input settings
    parser.add_argument(
        "--img_size",
        type=int,
        nargs=2,
        default=[640, 640],
        help="Input image size as two integers: width height (e.g. --img_size 640 640).",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="densenet121",
        choices=["mobilenetv1", "resnet50", "vgg16", "densenet121", "vit"],
        help="Backbone architecture to use.",
    )
    parser.add_argument(
        "--out_channel",
        type=int,
        default=64,
        help="Number of output channels for the FPN feature maps.",
    )
    parser.add_argument(
        "--use_pretrained",
        action="store_true",
        default=True,
        help="Load pretrained weights for the backbone.",
    )
    parser.add_argument(
        "--freeze_backbone",
        action="store_true",
        default=True,
        help="Freeze backbone parameters during training.",
    )
    parser.add_argument(
        "--no_freeze_backbone",
        action="store_false",
        dest="freeze_backbone",
        help="Do not freeze the backbone (override --freeze_backbone).",
    )

    # Training hyperparameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=config.DEFAULT_EPOCHS,
        help=f"Number of training epochs (default: {config.DEFAULT_EPOCHS}).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=config.DEFAULT_LR,
        help=f"Initial learning rate (default: {config.DEFAULT_LR}).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=config.DEFAULT_BATCH_SIZE,
        help=f"Batch size for training and validation (default: {config.DEFAULT_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=config.DEFAULT_WEIGHT_DECAY,
        help=f"Weight decay (L2) regularization coefficient (default: {config.DEFAULT_WEIGHT_DECAY}).",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default=config.DEFAULT_OPTIMIZER,
        choices=["ADAM", "SGD"],
        help="Optimizer to use: ADAM or SGD.",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default=config.DEFAULT_SCHEDULER,
        choices=[None, "ReduceLR", "OneCycle", "Cosine"],
        help="Learning rate scheduler: None, ReduceLR, OneCycle, or Cosine.",
    )
    parser.add_argument(
        "--clip_value",
        type=float,
        default=config.DEFAULT_CLIP_VALUE,
        help="Gradient clipping value (None to disable).",
    )
    parser.add_argument(
        "--grad_clip_mode",
        type=str,
        default=config.DEFAULT_GRAD_CLIP_MODE,
        choices=["Norm", "Value"],
        help="Gradient clipping mode: Norm or Value.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=config.DEFAULT_PATIENCE,
        help=f"EarlyStopping patience in epochs (default: {config.DEFAULT_PATIENCE}).",
    )

    # Loss weighting
    parser.add_argument(
        "--lambda_cls",
        type=float,
        default=config.LAMBDA_CLS,
        help="Weight for the orientation classification (focal) loss.",
    )
    parser.add_argument(
        "--lambda_face",
        type=float,
        default=config.LAMBDA_FACE,
        help="Weight for the face/no-face (BCE) loss.",
    )
    parser.add_argument(
        "--lambda_obb",
        type=float,
        default=config.LAMBDA_OBB,
        help="Weight for the oriented bounding box regression loss.",
    )
    parser.add_argument(
        "--lambda_rot",
        type=float,
        default=config.LAMBDA_ROT,
        help="Weight for the rotation angle regression loss.",
    )
    parser.add_argument(
        "--pos_iou_thr",
        type=float,
        default=config.POS_IOU_THRESH,
        help="Positive IoU threshold for anchor matching (default: 0.5).",
    )
    parser.add_argument(
        "--neg_iou_thr",
        type=float,
        default=config.NEG_IOU_THRESH,
        help="Negative IoU threshold (band-of-ignore) for anchor matching (default: 0.4).",
    )
    parser.add_argument(
        "--neg_samples_ratio",
        type=int,
        default=config.NEG_SAMPLES_RATIO,
        help="Hard negative mining ratio: negatives per positive (default: 3).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        nargs="+",
        default=config.ALPHA,
        help="Alpha class weights for Focal Loss, one per orientation class.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=config.GAMMA,
        help="Gamma (focusing parameter) for Focal Loss.",
    )

    # Data augmentation
    parser.add_argument(
        "--use_augmentation",
        action="store_true",
        default=True,
        help="Enable data augmentation in the training pipeline.",
    )
    parser.add_argument(
        "--no_augmentation",
        action="store_false",
        dest="use_augmentation",
        help="Disable data augmentation (override --use_augmentation).",
    )

    # Logging & tracking
    parser.add_argument(
        "--record_metrics",
        action="store_true",
        help="Log training metrics to Weights & Biases.",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=config.PROJECT_NAME,
        help=f"Weights & Biases project name (default: {config.PROJECT_NAME}).",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=config.RUN_NAME,
        help=f"Weights & Biases run name (default: {config.RUN_NAME}).",
    )

    # Inference
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to evaluate at inference time (train/val/test).",
    )
    parser.add_argument(
        "--conf_thres",
        type=float,
        default=config.CONF_THRESH,
        help=f"Face confidence threshold for inference (default: {config.CONF_THRESH}).",
    )
    parser.add_argument(
        "--iou_thres",
        type=float,
        default=config.IOU_THRESH,
        help=f"IoU threshold for rotated NMS (default: {config.IOU_THRESH}).",
    )
    parser.add_argument(
        "--class_thres",
        type=float,
        default=config.CLASS_THRESH,
        help=f"Orientation confidence threshold for inference (default: {config.CLASS_THRESH}).",
    )
    parser.add_argument(
        "--grid_rows",
        type=int,
        default=3,
        help="Number of rows in the qualitative mosaic grid.",
    )
    parser.add_argument(
        "--grid_cols",
        type=int,
        default=3,
        help="Number of columns in the qualitative mosaic grid.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    print("[INFO] Starting training and inference with args:", vars(args))

    # ------------------------------------------------------------------------
    # I. Output directory structure
    # ------------------------------------------------------------------------
    output_dir = Path(args.run_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = output_dir / "checkpoint.pt"
    csv_path = output_dir / f"{args.run_name}.csv"
    config_path = output_dir / f"{args.run_name}.yaml"
    figures_dir = output_dir / "figures"
    grids_dir = output_dir / "dataset_grids"
    predictions_dir = output_dir / "predictions"
    anchor_preview_path = output_dir / "anchors_preview.jpg"
    inference_preview = output_dir / "training_grids"

    inference_preview.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)
    grids_dir.mkdir(exist_ok=True)
    predictions_dir.mkdir(exist_ok=True)

    # Save full config to YAML
    with open(config_path, "w") as f:
        yaml.dump(vars(args), f)
    print(f"[INFO] Saved config to {config_path}")

    # ------------------------------------------------------------------------
    # II. Setup
    # ------------------------------------------------------------------------
    set_seed(42)
    device = get_default_device()
    print(f"[INFO] Using device: {device}")

    norm_mean = config.IMAGENET_MEAN if args.freeze_backbone else config.MEAN
    norm_std = config.IMAGENET_STD if args.freeze_backbone else config.STD
    img_size = tuple(args.img_size)

    train_transform = config.get_train_transform(
        img_size, args.use_augmentation, mean=norm_mean, std=norm_std
    )
    val_transform = config.get_val_transform(img_size, mean=norm_mean, std=norm_std)

    # ------------------------------------------------------------------------
    # III. Datasets and loaders
    # ------------------------------------------------------------------------
    train_dataset = BabyFacesDataset(
        args.root_dir, split="train", transform=train_transform
    )
    val_dataset = BabyFacesDataset(args.root_dir, split="val", transform=val_transform)

    print(
        f"[INFO] Loaded {len(train_dataset)} training and {len(val_dataset)} validation samples."
    )

    # Optional: visualize datasets
    visualize_and_save_dataset_in_script(
        train_dataset, "train", grids_dir, num_images=9
    )
    visualize_and_save_dataset_in_script(val_dataset, "val", grids_dir, num_images=9)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=custom_collate,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=custom_collate,
        num_workers=4,
        pin_memory=True,
    )

    # ------------------------------------------------------------------------
    # IV. Model and loss setup
    # ------------------------------------------------------------------------
    model = RetinaBabyFace(
        args.backbone, args.out_channel, pretrained=args.use_pretrained
    ).to(device)

    if args.freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad = False
        for m in model.backbone.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False

    reset_heads(model)
    summary(
        model,
        input_size=(1, 3, img_size[1], img_size[0]),
        col_names=["input_size", "output_size", "num_params", "trainable"],
        row_settings=["var_names"],
        col_width=20,
        depth=2,
        device=device.type,
    )

    multitask_loss = MultiTaskLoss(
        args.lambda_cls,
        args.lambda_obb,
        args.lambda_rot,
        args.lambda_face,
        args.pos_iou_thr,
        args.neg_iou_thr,
        args.alpha,
        args.gamma,
        args.neg_samples_ratio,
    )
    earlystopping = EarlyStopping(
        args.patience, verbose=True, delta=0.001, path=ckpt_path
    )

    # ------------------------------------------------------------------------
    # V. Training
    # ------------------------------------------------------------------------
    print("[INFO] Starting training...")
    train(
        model,
        train_loader,
        val_loader,
        multitask_loss,
        args.optimizer,
        args.weight_decay,
        args.lr,
        args.epochs,
        device,
        earlystopping,
        args.scheduler,
        args.clip_value,
        args.grad_clip_mode,
        args.record_metrics,
        args.project,
        args.run_name,
        config.SCALE_FACTORS,
        config.RATIO_FACTORS,
        config.PRECOMPUTED_OBB_STATS,
        csv_path=csv_path,
        anchor_preview_path=anchor_preview_path,
        inference_preview=inference_preview,
    )

    print("\n[INFO] Training completed!")

    # ------------------------------------------------------------------------
    # VI. Inference
    # ------------------------------------------------------------------------
    print("[INFO] Starting inference...")

    test_dataset = BabyFacesDataset(
        args.root_dir, split=args.split, transform=val_transform
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=custom_collate,
        num_workers=4,
        pin_memory=True,
    )

    visualize_and_save_dataset_in_script(test_dataset, "test", grids_dir, num_images=9)

    labels_map = {
        0: "3/4 Leftside",
        1: "3/4 Rightside",
        2: "Frontal",
        3: "Left Profile",
        4: "Right Profile",
    }

    # Reload model for inference
    trained_model = RetinaBabyFace(
        args.backbone, args.out_channel, pretrained=args.use_pretrained
    ).to(device)
    print(f"[INFO] Loading weights from: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    trained_model.load_state_dict(
        state["model_state_dict"] if "model_state_dict" in state else state
    )
    trained_model.eval()

    figures = inference(
        trained_model,
        checkpoint_path=ckpt_path,
        test_loader=test_loader,
        output_dir=predictions_dir,
        device=device,
        labels_map=labels_map,
        scale_factors=config.SCALE_FACTORS,
        ratio_factors=config.RATIO_FACTORS,
        obb_stats_by_size=config.PRECOMPUTED_OBB_STATS,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        class_thres=args.class_thres,
        grid_shape=(args.grid_rows, args.grid_cols),
        mean=norm_mean,
        std=norm_std,
    )

    # Save all figures
    figures["pr_figure"].savefig(figures_dir / "precision_recall.png", dpi=150)
    figures["confusion_figure"].savefig(figures_dir / "confusion_matrix.png", dpi=150)
    figures["grid_figure"].savefig(figures_dir / "grid_examples.png", dpi=150)
    figures["iou_boxplot_figure"].savefig(figures_dir / "iou_boxplot.png", dpi=150)
    figures["angle_boxplot_figure"].savefig(figures_dir / "angle_boxplot.png", dpi=150)
    figures["f1_threshold_figure"].savefig(figures_dir / "f1_threshold.png", dpi=150)

    print(f"[INFO] All figures saved to {figures_dir}")
    print(f"[INFO] All predictions saved to {predictions_dir}")
    print(f"[INFO] All done! Check {output_dir} for results.")

    # # Create a GIF of the training process
    # create_training_gif(image_folder=inference_preview, output_path=output_dir / "training.gif")
    # print(f"[INFO] Training GIF saved to {output_dir / 'training.gif'}")
    # print(f"[INFO] All done! Check {output_dir} for results.")


if __name__ == "__main__":
    main()
