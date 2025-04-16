# This script is used to train the RetinaBabyFace model on the BabyFace dataset.
# It includes data loading, augmentation, model definition, and training loop.
# The script uses PyTorch and torchvision for model training and data handling.
# The RetinaBabyFace model is a custom architecture designed for face detection and recognition tasks.

import argparse
import os
import sys

# Adding the root directory to the system path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
print(f"[INFO] Adding {ROOT_DIR} to sys.path")
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

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
    parser.add_argument("--root_dir", type=str, required=True, help="Path to dataset root directory")
    parser.add_argument("--pretrain_path", type=str, required=True, help="Path to MobileNet pretrained weights")

    # Training hyperparams
    parser.add_argument("--epochs", type=int, default=config.DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=config.DEFAULT_LR)
    parser.add_argument("--batch_size", type=int, default=config.DEFAULT_BATCH_SIZE)
    parser.add_argument("--weight_decay", type=float, default=config.DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--optimizer", type=str, default=config.DEFAULT_OPTIMIZER)
    parser.add_argument("--scheduler", type=str, default=config.DEFAULT_SCHEDULER)
    parser.add_argument("--clip_value", type=float, default=config.DEFAULT_CLIP_VALUE)
    parser.add_argument("--grad_clip_mode", type=str, default=config.DEFAULT_GRAD_CLIP_MODE)
    parser.add_argument("--patience", type=int, default=config.DEFAULT_PATIENCE)

    # WandB
    parser.add_argument("--record_metrics", action="store_true")
    parser.add_argument("--project", type=str, default=config.PROJECT_NAME)
    parser.add_argument("--run_name", type=str, default=config.RUN_NAME)

    return parser.parse_args()

def main():
    args = parse_args()
    print("[INFO] Starting training script with args:", vars(args))

    set_seed(42)
    device = torch.device("cpu") #get_default_device()
    print(f"[INFO] Using device: {device}")

    print("[INFO] Loading datasets...")
    train_dataset = BabyFacesDataset(
        root_dir=args.root_dir, split="train", transform=config.get_train_transform()
    )
    val_dataset = BabyFacesDataset(
        root_dir=args.root_dir, split="val", transform=config.get_val_transform()
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
    backbone = MobileNetV1()
    return_layers = {"stage1": "0", "stage2": "1", "stage3": "2"}
    model = RetinaBabyFace(
        backbone=backbone,
        return_layers=return_layers,
        in_channel=64,
        out_channel=64,
        pretrain_path=args.pretrain_path,
    ).to(device)
    
    print("[INFO] Model summary:")
    summary(model, input_size=(1, 3, 640, 640), col_names=["input_size", "output_size", "num_params", "trainable"], row_settings=["var_names"], col_width=20, depth=2)

    multitask_loss = MultiTaskLoss(lambda_class=1.0, lambda_obb=1.0, lambda_rot=1.0)
    earlystopping = EarlyStopping(
        patience=args.patience,
        verbose=True,
        delta=0.001,
        path="checkpoint.pt",
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
        base_size=config.BASE_SIZE,
        base_ratio=config.BASE_RATIO,
        scale_factors=config.SCALE_FACTORS,
        ratio_factors=config.RATIO_FACTORS,
    )

    print("\n[INFO] Training completed!")


if __name__ == "__main__":
    main()
