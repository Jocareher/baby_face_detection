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
from utils.visualize import visualize_and_save_dataset_in_script
import config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate RetinaBabyFace model"
    )

    # Dataset and paths
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Path to the dataset root directory.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="checkpoint.pt",
        help="Path to save/load the model checkpoint.",
    )
    parser.add_argument(
        "--inference_results",
        type=str,
        default="inference_results",
        help="Directory to save the inference plots.",
    )
    parser.add_argument(
        "--predictions_dir",
        type=str,
        default="predictions",
        help="Directory to save per-image prediction visualizations.",
    )

    # Model and image input
    parser.add_argument(
        "--img_size",
        type=int,
        nargs=2,
        default=[640, 640],
        help="Input image size as (width height).",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="densenet121",
        choices=["mobilenetv1", "resnet50", "vgg16", "densenet121", "vit"],
        help="Backbone architecture.",
    )
    parser.add_argument(
        "--out_channel",
        type=int,
        default=64,
        help="Number of output channels for feature maps.",
    )
    parser.add_argument(
        "--use_pretrained",
        action="store_true",
        default=True,
        help="Use pretrained weights for the backbone.",
    )
    parser.add_argument(
        "--freeze_backbone",
        action="store_true",
        default=True,
        help="Freeze the backbone during training.",
    )
    parser.add_argument(
        "--no_freeze_backbone",
        action="store_false",
        dest="freeze_backbone",
        help="Do not freeze the backbone.",
    )

    # Training hyperparameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=config.DEFAULT_EPOCHS,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--lr", type=float, default=config.DEFAULT_LR, help="Initial learning rate."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=config.DEFAULT_BATCH_SIZE,
        help="Batch size for training and validation.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=config.DEFAULT_WEIGHT_DECAY,
        help="Weight decay for optimizer.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default=config.DEFAULT_OPTIMIZER,
        choices=["ADAM", "SGD"],
        help="Optimizer type.",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default=config.DEFAULT_SCHEDULER,
        choices=[None, "ReduceLR", "OneCycle", "Cosine"],
        help="Learning rate scheduler.",
    )
    parser.add_argument(
        "--clip_value",
        type=float,
        default=config.DEFAULT_CLIP_VALUE,
        help="Max norm for gradient clipping.",
    )
    parser.add_argument(
        "--grad_clip_mode",
        type=str,
        default=config.DEFAULT_GRAD_CLIP_MODE,
        choices=["Norm", "Value"],
        help="Gradient clipping strategy.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=config.DEFAULT_PATIENCE,
        help="Early stopping patience (epochs).",
    )

    # Loss weighting
    parser.add_argument(
        "--lambda_cls",
        type=float,
        default=config.LAMBDA_CLS,
        help="Weight for classification loss.",
    )
    parser.add_argument(
        "--lambda_obb",
        type=float,
        default=config.LAMBDA_OBB,
        help="Weight for OBB regression loss.",
    )
    parser.add_argument(
        "--lambda_rot",
        type=float,
        default=config.LAMBDA_ROT,
        help="Weight for angle regression loss.",
    )

    # Data augmentation
    parser.add_argument(
        "--use_augmentation",
        action="store_true",
        default=True,
        help="Apply data augmentation during training.",
    )
    parser.add_argument(
        "--no_augmentation",
        action="store_false",
        dest="use_augmentation",
        help="Disable data augmentation.",
    )

    # Logging & tracking
    parser.add_argument(
        "--record_metrics", action="store_true", help="Enable logging to WandB."
    )
    parser.add_argument(
        "--project", type=str, default=config.PROJECT_NAME, help="WandB project name."
    )
    parser.add_argument(
        "--run_name", type=str, default=config.RUN_NAME, help="WandB run name."
    )

    # Inference configuration
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to evaluate: 'test' or 'val'.",
    )
    parser.add_argument(
        "--conf_thres",
        type=float,
        default=0.25,
        help="Confidence threshold for predictions.",
    )
    parser.add_argument(
        "--iou_thres", type=float, default=0.5, help="IoU threshold for evaluation."
    )
    parser.add_argument(
        "--grid_rows",
        type=int,
        default=3,
        help="Number of rows for visualization grid.",
    )
    parser.add_argument(
        "--grid_cols",
        type=int,
        default=3,
        help="Number of columns for visualization grid.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    print("[INFO] Starting training and inference with args:", vars(args))

    # Save config
    with open(f"{args.run_name}.yaml", "w") as f:
        yaml.dump(vars(args), f)

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

    train_dataset = BabyFacesDataset(
        args.root_dir, split="train", transform=train_transform
    )
    val_dataset = BabyFacesDataset(args.root_dir, split="val", transform=val_transform)

    print(
        f"[INFO] Loaded {len(train_dataset)} training and {len(val_dataset)} validation samples."
    )

    print("[INFO] Visualizing training dataset...")
    visualize_and_save_dataset_in_script(
        train_dataset, "train", args.inference_results, num_images=9
    )
    print("[INFO] Visualizing validation dataset...")
    visualize_and_save_dataset_in_script(
        val_dataset, "val", args.inference_results, num_images=9
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

    multitask_loss = MultiTaskLoss(args.lambda_cls, args.lambda_obb, args.lambda_rot)
    earlystopping = EarlyStopping(
        args.patience, verbose=True, delta=0.001, path=args.checkpoint_path
    )

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
    )

    print("\n[INFO] Training completed!")

    # INFERENCE
    print("[INFO] Starting inference...")

    test_dataset = BabyFacesDataset(
        args.root_dir, split=args.split, transform=val_transform
    )

    print(f"[INFO] Loaded {len(test_dataset)} samples from split: {args.split}")
    print("[INFO] Visualizing test dataset...")
    visualize_and_save_dataset_in_script(
        test_dataset, "test", args.inference_results, num_images=9
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=custom_collate,
        num_workers=4,
        pin_memory=True,
    )

    labels_map = {
        0: "3/4 Leftside",
        1: "3/4 Rightside",
        2: "Frontal",
        3: "Left Profile",
        4: "Right Profile",
    }

    trained_model = RetinaBabyFace(
        backbone_name=args.backbone,
        out_channel=args.out_channel,
        pretrained=args.use_pretrained,
    ).to(device)

    print(f"[INFO] Loading weights from: {args.checkpoint_path}")
    state = torch.load(args.checkpoint_path, map_location=device)
    if "model_state_dict" in state:
        trained_model.load_state_dict(state["model_state_dict"])
    else:
        trained_model.load_state_dict(state)

    trained_model.eval()

    figures = inference(
        trained_model,
        args.checkpoint_path,
        test_loader,
        args.predictions_dir,
        device,
        labels_map,
        config.SCALE_FACTORS,
        config.RATIO_FACTORS,
        config.PRECOMPUTED_OBB_STATS,
        args.conf_thres,
        args.iou_thres,
        (args.grid_rows, args.grid_cols),
        mean=norm_mean,
        std=norm_std,
    )

    os.makedirs(args.inference_results, exist_ok=True)
    figures["pr_figure"].savefig(
        os.path.join(args.inference_results, "precision_recall.png"), dpi=150
    )
    figures["confusion_figure"].savefig(
        os.path.join(args.inference_results, "confusion_matrix.png"), dpi=150
    )
    figures["grid_figure"].savefig(
        os.path.join(args.inference_results, "grid_examples.png"), dpi=150
    )
    figures["iou_boxplot_figure"].savefig(
        os.path.join(args.inference_results, "iou_boxplot_figure.png"), dpi=150
    )
    figures["angle_boxplot_figure"].savefig(
        os.path.join(args.inference_results, "angle_boxplot_figure.png"), dpi=150
    )
    figures["f1_threshold_figure"].savefig(
        os.path.join(args.inference_results, "f1_threshold_figure.png"), dpi=150
    )

    print("[INFO] Inference and evaluation completed.")
    print(f"[INFO] All figures saved to: {args.inference_results}")


if __name__ == "__main__":
    main()
