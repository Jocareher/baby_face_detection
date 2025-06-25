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

from data_setup.dataset import BabyFacesDataset, make_balanced_sampler
from data_setup.collate import custom_collate
from models.retinababyface import RetinaBabyFace, reset_heads, set_backbone_frozen
from utils.helpers import set_seed, get_default_device
from engine.train import train, EarlyStopping, load_checkpoint_for_resuming
from engine.inference import inference, plot_training_curves_from_csv
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

    parser.add_argument(
        "--balanced_sampler",
        action="store_true",
        default=False,
        help="Use a balanced sampler for the training dataset.",
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
        choices=[
            "mobilenetv1",
            "resnet50",
            "vgg16",
            "densenet121",
            "vit",
            "vggface2",
            "arcface",
        ],
        help="Backbone architecture to use.",
    )
    parser.add_argument(
        "--out_channel",
        type=int,
        default=config.DEFAULT_OUT_CHANNELS,
        help="Number of output channels for the FPN feature maps.",
    )
    parser.add_argument(
        "--use_pretrained",
        action="store_true",
        default=True,
        help="Load pretrained weights for the backbone.",
    )

    parser.add_argument(
        "--backbone_mode",
        type=str,
        default="feature_extractor",
        choices=["feature_extractor", "fine_tuning", "train_all"],
        help="How to treat the backbone parameters during training.",
    )

    parser.add_argument(
        "--resume_training",
        type=str,
        default=None,
        help="Path to a checkpoint file to resume training from (e.g. run_name/checkpoint.pt).",
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
        choices=["ADAM", "SGD", "ADAMW", "RAdam"],
        help="Optimizer to use: ADAM, ADAMW, RAdam, or SGD.",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default=config.DEFAULT_SCHEDULER,
        choices=["ReduceLR", "OneCycle", "Cosine"],
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
        "--obb_loss_type",
        type=str,
        default=config.OBB_LOSS_TYPE,
        choices=["l1", "smooth_l1"],
        help="Type of loss to use for oriented bounding box regression (default: L1)",
    )
    parser.add_argument(
        "--rot_loss_type",
        type=str,
        default=config.ROT_LOSS_TYPE,
        choices=["cosine", "vector"],
        help="Type of loss to use for rotation angle regression (default: cosine)",
    )
    parser.add_argument(
        "--cls_loss_type",
        type=str,
        default=config.CLS_LOSS_TYPE,
        choices=["focal", "ls"],
        help="Type of loss to use for orientation classification (default: focal)",
    )
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
        "--face_pos_weight",
        type=float,
        default=config.FACE_POS_WEIGHT,
        help="Weight for positive face samples in the loss function (default: 2.0).",
    )
    parser.add_argument(
        "--pos_iou_thr_1",
        type=float,
        default=config.POS_IOU_THRESH_1,
        help="Positive IoU threshold for 1-stage anchor matching (default: 0.7).",
    )
    parser.add_argument(
        "--neg_iou_thr_1",
        type=float,
        default=config.NEG_IOU_THRESH_1,
        help="Negative IoU threshold (band-of-ignore) for 1-stage anchor matching (default: 0.3).",
    )
    parser.add_argument(
        "--pos_iou_thr_2",
        type=float,
        default=config.POS_IOU_THRESH_2,
        help="Positive IoU threshold for 2-stage anchor matching (default: 0.5).",
    )
    parser.add_argument(
        "--neg_iou_thr_2",
        type=float,
        default=config.NEG_IOU_THRESH_2,
        help="Negative IoU threshold (band-of-ignore) for 2-stage anchor matching (default: 0.4).",
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
        "--face_thres",
        type=float,
        default=config.FACE_THRESH,
        help=f"Face confidence threshold for inference (default: {config.FACE_THRESH}).",
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
    # Set random seed for reproducibility
    set_seed(42)

    # Get the default device (CPU or GPU)
    device = get_default_device()
    print(f"[INFO] Using device: {device}")

    # Determine normalization statistics based on backbone mode
    if args.backbone_mode == "train_all":
        norm_mean, norm_std = config.MEAN, config.STD
    else:
        norm_mean, norm_std = config.IMAGENET_MEAN, config.IMAGENET_STD

    # Define image size tuple
    img_size = tuple(args.img_size)

    # Configure data augmentation and normalization transforms for training and validation
    train_transform = config.get_train_transform(
        img_size, args.use_augmentation, mean=norm_mean, std=norm_std
    )
    val_transform = config.get_val_transform(img_size, mean=norm_mean, std=norm_std)

    # ------------------------------------------------------------------------
    # III. Datasets and loaders
    # ------------------------------------------------------------------------

    # Load training and validation datasets
    train_dataset = BabyFacesDataset(
        args.root_dir, split="train", transform=train_transform
    )
    val_dataset = BabyFacesDataset(args.root_dir, split="val", transform=val_transform)

    print(
        f"[INFO] Loaded {len(train_dataset)} training and {len(val_dataset)} validation samples."
    )

    # Optional: visualize datasets and save sample grids
    visualize_and_save_dataset_in_script(
        train_dataset, "train", grids_dir, num_images=9
    )
    visualize_and_save_dataset_in_script(val_dataset, "val", grids_dir, num_images=9)

    if args.balanced_sampler:
        sampler = make_balanced_sampler(train_dataset)
        # Create data loaders for training and validation datasets
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            collate_fn=custom_collate,
            num_workers=4,
            pin_memory=True,
        )
        print(
            f"[INFO] Using balanced sampler for training dataset with {len(sampler)} samples."
        )

    else:
        # Create data loaders for training and validation datasets
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

    # Initialize the RetinaBabyFace model with the specified backbone and output channels
    model = RetinaBabyFace(
        args.backbone, args.out_channel, pretrained=args.use_pretrained
    ).to(device)

    # Configure backbone freezing or fine-tuning based on the specified mode
    set_backbone_frozen(
        model,
        mode=args.backbone_mode,
    )

    # Apply Kaiming initialization to the model heads
    reset_heads(model)

    # Save and print model summary
    with open(output_dir / "model_summary.txt", "w") as f:
        f.write(
            str(
                summary(
                    model,
                    input_size=(1, 3, img_size[1], img_size[0]),
                    col_names=["input_size", "output_size", "num_params", "trainable"],
                    row_settings=["var_names"],
                    col_width=20,
                    depth=2,
                    device=device.type,
                )
            )
        )
    print(f"[INFO] Model summary saved to {output_dir / 'model_summary.txt'}")

    # If resuming training from a checkpoint
    if args.resume_training:
        load_checkpoint_for_resuming(model, args.resume_training, device)

    # If using OneCycleLR scheduler, warn about resuming training without scheduler state
    if args.scheduler == "OneCycle" and args.resume_training:
        print(
            "[WARNING] OneCycleLR is not recommended when resuming training without scheduler state. Consider using ReduceLR or Cosine."
        )

    # Compile the model if using CUDA and PyTorch 2.0 or later
    if device.type == "cuda":
        print("[INFO] Compiling model with torch.compile...")
        model = torch.compile(
            model
        )  # torch.compile is only available in PyTorch 2.0 and later
        print("[INFO] Model compilation complete.")

    # Initialize the multi-task loss function with specified weights and thresholds
    multitask_loss = MultiTaskLoss(
        args.obb_loss_type,
        args.rot_loss_type,
        args.cls_loss_type,
        args.lambda_cls,
        args.lambda_obb,
        args.lambda_rot,
        args.lambda_face,
        args.pos_iou_thr_1,
        args.neg_iou_thr_1,
        args.pos_iou_thr_2,
        args.neg_iou_thr_2,
        args.alpha,
        args.gamma,
        args.neg_samples_ratio,
        args.face_pos_weight,
    )

    # Initialize early stopping mechanism
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
        face_thres=args.face_thres,
        iou_thres=args.iou_thres,
        class_thres=args.class_thres,
        csv_path=csv_path,
        anchor_preview_path=anchor_preview_path,
        inference_preview=inference_preview,
    )

    print("\n[INFO] Training completed!")

    # ------------------------------------------------------------------------
    # VI. Inference
    # ------------------------------------------------------------------------

    print("[INFO] Starting inference...")

    # Load test dataset and create data loader
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

    # Visualize and save test dataset samples
    visualize_and_save_dataset_in_script(test_dataset, "test", grids_dir, num_images=9)

    # Define label mapping for inference
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
    # Load the saved model weights from the checkpoint file
    raw_weights = torch.load(ckpt_path, map_location=device)

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
    trained_model.load_state_dict(state_dict)

    # Set the model to evaluation mode
    trained_model.eval()

    # Perform inference and save results
    figures = inference(
        trained_model,
        test_loader=test_loader,
        output_dir=predictions_dir,
        device=device,
        labels_map=labels_map,
        scale_factors=config.SCALE_FACTORS,
        ratio_factors=config.RATIO_FACTORS,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        class_thres=args.class_thres,
        grid_shape=(args.grid_rows, args.grid_cols),
        mean=norm_mean,
        std=norm_std,
    )

    # Save all figures generated during inference
    figures["pr_figure"].savefig(figures_dir / "precision_recall.png", dpi=150)
    figures["confusion_figure"].savefig(figures_dir / "confusion_matrix.png", dpi=150)
    figures["grid_figure"].savefig(figures_dir / "grid_examples.png", dpi=150)
    figures["iou_boxplot_figure"].savefig(figures_dir / "iou_boxplot.png", dpi=150)
    figures["angle_boxplot_figure"].savefig(figures_dir / "angle_boxplot.png", dpi=150)
    figures["f1_threshold_figure"].savefig(figures_dir / "f1_threshold.png", dpi=150)

    # Plot training curves from the CSV file
    print(f"[INFO] Plotting training curves from {csv_path}")
    plot_training_curves_from_csv(csv_path, output_dir)

    print(f"[INFO] All figures saved to {figures_dir}")
    print(f"[INFO] All predictions saved to {predictions_dir}")
    print(f"[INFO] All done! Check {output_dir} for results.")

    # # Create a GIF of the training process
    # create_training_gif(image_folder=inference_preview, output_path=output_dir / "training.gif")
    # print(f"[INFO] Training GIF saved to {output_dir / 'training.gif'}")
    # print(f"[INFO] All done! Check {output_dir} for results.")


if __name__ == "__main__":
    main()
