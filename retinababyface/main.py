# This script is used to train the RetinaBabyFace model on the BabyFace dataset.
# It includes data loading, augmentation, model definition, and training loop.
# The script uses PyTorch and torchvision for model training and data handling.
# The RetinaBabyFace model is a custom architecture designed for face detection and recognition tasks.

import argparse
import os
import sys

# Adding the root directory to the system path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
    print(f"[INFO] Adding {ROOT_DIR} to sys.path")


import torch
from torch.utils.data import DataLoader
from torchinfo import summary

from data_setup.dataset import BabyFacesDataset
from data_setup.collate import custom_collate
from models.mobilenet import MobileNetV1
from models.retinababyface import RetinaBabyFace
from utils.helpers import set_seed, get_default_device
from engine.train import train, EarlyStopping
from loss.losses import MultiTaskLoss
import config


def parse_args():
    parser = argparse.ArgumentParser(description="Train RetinaBabyFace Model")

    # Dataset and paths
    parser.add_argument(
        "--root_dir", 
        type=str, 
        required=True, 
        help="Path to dataset root directory"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="checkpoint.pt",
        help="Path to save model checkpoints",
    )

    # Image parameters
    parser.add_argument(
        "--img_size",
        type=int,
        nargs=2,
        default=[640, 640],
        help="Image size as width height (default: 640 640)",
    )

    # Model architecture
    parser.add_argument(
        "--use_pretrained",
        action="store_true",
        default=True,
        help="Use pretrained weights (default: True)",
    )
    parser.add_argument(
        "--freeze_backbone",
        action="store_true",
        default=True,
        help="Freeze backbone weights during training (default: True)",
    )
    parser.add_argument(
        "--no_freeze_backbone",
        action="store_false",
        dest="freeze_backbone",
        help="Disable backbone weight freezing",
    )
    parser.add_argument(
        "--out_channel", 
        type=int, 
        default=64, 
        help="Number of output channels for FPN"
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet50",
        choices=["mobilenetv1", "resnet50", "vgg16", "densenet121", "vit"],
        help="Backbone architecture to use",
    )

    # Training hyperparameters (using config defaults)
    parser.add_argument("--epochs", type=int, default=config.DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=config.DEFAULT_LR)
    parser.add_argument("--batch_size", type=int, default=config.DEFAULT_BATCH_SIZE)
    parser.add_argument("--weight_decay", type=float, default=config.DEFAULT_WEIGHT_DECAY)
    parser.add_argument(
        "--optimizer",
        type=str,
        default=config.DEFAULT_OPTIMIZER,
        choices=["ADAM", "SGD"],
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default=config.DEFAULT_SCHEDULER,
        choices=[None, "ReduceLR", "OneCycle", "Cosine"],
    )
    parser.add_argument("--clip_value", type=float, default=config.DEFAULT_CLIP_VALUE)
    parser.add_argument(
        "--grad_clip_mode",
        type=str,
        default=config.DEFAULT_GRAD_CLIP_MODE,
        choices=["Norm", "Value"],
    )
    parser.add_argument("--patience", type=int, default=config.DEFAULT_PATIENCE)

    # Loss function parameters
    parser.add_argument(
        "--lambda_cls", type=float, default=1.0, help="Weight for classification loss"
    )
    parser.add_argument(
        "--lambda_obb", type=float, default=1.0, help="Weight for OBB regression loss"
    )
    parser.add_argument(
        "--lambda_rot", type=float, default=1.0, help="Weight for rotation angle loss"
    )

    # Data augmentation
    parser.add_argument(
        "--use_augmentation",
        action="store_true",
        default=True,
        help="Enable data augmentation during training (default: True)",
    )
    parser.add_argument(
        "--no_augmentation",
        action="store_false",
        dest="use_augmentation",
        help="Disable data augmentation",
    )

    # WandB
    parser.add_argument(
        "--record_metrics", action="store_true", help="Enable WandB logging"
    )
    parser.add_argument(
        "--project", type=str, default=config.PROJECT_NAME, help="WandB project name"
    )
    parser.add_argument(
        "--run_name", type=str, default=config.RUN_NAME, help="WandB run name"
    )

    return parser.parse_args()

def main():
    args = parse_args()
    print("[INFO] Starting training script with args:", vars(args))

    set_seed(42)
    device = get_default_device()
    print(f"[INFO] Using device: {device}")

    print("[INFO] Loading datasets...")
    img_size = tuple(args.img_size)  # Convert to (width, height)

    train_transform = config.get_train_transform(img_size, args.use_augmentation)
    val_transform = config.get_val_transform(img_size)

    train_dataset = BabyFacesDataset(
        root_dir=args.root_dir, split="train", transform=train_transform
    )
    val_dataset = BabyFacesDataset(
        root_dir=args.root_dir, split="val", transform=val_transform
    )

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

    print("[INFO] Building model...")
    model = RetinaBabyFace(
        backbone_name=args.backbone,
        out_channel=args.out_channel,
        pretrained=args.use_pretrained,
        freeze_backbone=args.freeze_backbone,
    ).to(device)

    print("[INFO] Model summary:")
    summary(
        model,
        input_size=(1, 3, img_size[1], img_size[0]),  # (channels, height, width)
        col_names=["input_size", "output_size", "num_params", "trainable"],
        row_settings=["var_names"],
        col_width=20,
        depth=2,
    )

    multitask_loss = MultiTaskLoss(
        lambda_cls=args.lambda_cls,
        lambda_obb=args.lambda_obb,
        lambda_rot=args.lambda_rot,
    )

    earlystopping = EarlyStopping(
        patience=args.patience,
        verbose=True,
        delta=0.001,
        path=args.checkpoint_path,
        use_kfold=False,
    )

    print("[INFO] Starting training...")
    results = train(
        model=model,
        train_dataloader=train_loader,
        test_dataloader=val_loader,
        loss_fn=multitask_loss,
        which_optimizer=args.optimizer,
        weight_decay=args.weight_decay,
        learning_rate=args.lr,
        epochs=args.epochs,
        device=device,
        early_stopping=earlystopping,
        which_scheduler=args.scheduler,
        clip_value=args.clip_value,
        grad_clip_mode=args.grad_clip_mode,
        record_metrics=args.record_metrics,
        project=args.project,
        run_name=args.run_name,
        scale_factors=config.SCALE_FACTORS,  # From config.py
        ratio_factors=config.RATIO_FACTORS,  # From config.py
        obb_stats_by_size=config.PRECOMPUTED_OBB_STATS  # From config.py
    )

    print("\n[INFO] Training completed!")

if __name__ == "__main__":
    main()