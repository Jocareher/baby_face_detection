# This script is used to train the RetinaBabyFace model on the BabyFace dataset.
# It includes data loading, augmentation, model definition, and training loop.
# The script uses PyTorch and torchvision for model training and data handling.
# The RetinaBabyFace model is a custom architecture designed for face detection and recognition tasks.

import argparse
import os
import sys

# Adding the root directory to the system path to import modules from the parent directory
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
print(f"[INFO] Adding {ROOT_DIR} to sys.path")
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from data_setup.dataset import BabyFacesDataset
from data_setup.augmentations import (
    RandomHorizontalFlipOBB,
    RandomRotateOBB,
    RandomScaleTranslateOBB,
    ColorJitterOBB,
    RandomNoiseOBB,
    RandomBlurOBB,
    RandomOcclusionOBB,
    Resize,
    ToTensorNormalize,
)
from data_setup.collate import custom_collate
from models.mobilenet import MobileNetV1
from models.retinababyface import RetinaBabyFace
from utils.helpers import set_seed, get_default_device
from engine.train import train, EarlyStopping
from loss.losses import MultiTaskLoss


def parse_args():
    parser = argparse.ArgumentParser(description="Train RetinaBabyFace Model")

    # Dataset and paths
    parser.add_argument(
        "--root_dir", type=str, required=True, help="Path to dataset root directory"
    )
    parser.add_argument(
        "--pretrain_path",
        type=str,
        required=True,
        help="Path to MobileNet pretrained weights",
    )

    # Training hyperparams
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--optimizer", type=str, default="ADAM")
    parser.add_argument("--scheduler", type=str, default=None)
    parser.add_argument("--clip_value", type=float, default=None)
    parser.add_argument("--grad_clip_mode", type=str, default="Norm")
    parser.add_argument("--patience", type=int, default=5)

    # WandB
    parser.add_argument("--record_metrics", action="store_true")
    parser.add_argument("--project", type=str, default="RetinaBabyFace")
    parser.add_argument("--run_name", type=str, default="run_1")

    return parser.parse_args()


def main():
    args = parse_args()

    print("[INFO] Starting training script with args:", vars(args))

    set_seed(42)
    device = get_default_device()
    print(f"[INFO] Using device: {device}")

    train_transform = transforms.Compose(
        [
            RandomHorizontalFlipOBB(prob=0.5),
            RandomRotateOBB(max_angle=180, prob=0.3),
            RandomScaleTranslateOBB(
                scale_range=(0.8, 1.1), translate_range=(-0.2, 0.2), prob=0.3
            ),
            ColorJitterOBB(brightness=0.2, contrast=0.2, saturation=0.2, prob=0.5),
            RandomNoiseOBB(std=10, prob=0.5),
            RandomBlurOBB(ksize=(5, 5), prob=0.3),
            RandomOcclusionOBB(max_size_ratio=0.3, prob=0.3),
            Resize((640, 640)),
            ToTensorNormalize(
                mean=(0.6427, 0.5918, 0.5525), std=(0.2812, 0.2825, 0.3036)
            ),
        ]
    )

    val_transform = transforms.Compose(
        [
            Resize((640, 640)),
            ToTensorNormalize(
                mean=(0.6427, 0.5918, 0.5525), std=(0.2812, 0.2825, 0.3036)
            ),
        ]
    )

    print("[INFO] Loading datasets...")
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
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=custom_collate,
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
    )

    print("[INFO] Training completed!")


if __name__ == "__main__":
    main()
