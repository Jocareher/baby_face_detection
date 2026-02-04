# This script is used to train the NewBORN model on the BabyFace dataset.
# It includes data loading, augmentation, model definition, and training loop.
# The script uses PyTorch and torchvision for model training and data handling.
# The NewBORN model is a custom architecture designed for face detection and recognition tasks.

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
from data_setup.augmentations import (
    collect_deg_by_class_from_dataset,
    plot_histograms_split,
    build_bin_weights_from_degrees,
    save_counts_csv,
)
from data_setup.collate import custom_collate
from data_setup.samplers import make_stratified_batch_sampler, make_weighted_sampler
from models.newborn import NewBORN, reset_heads, set_backbone_frozen
from utils.helpers import set_seed, get_default_device, seed_worker
from engine.train import train, EarlyStopping, load_checkpoint_for_resuming
from engine.inference import inference
from loss.losses import MultiTaskLoss
from utils.visualize import (
    visualize_and_save_dataset_in_script,
    plot_training_curves_from_csv,
)
from utils.repro import save_reproducibility_metadata
import config


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate NewBORN model")

    # Dataset
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Path to the dataset root directory (containing train/val/test subfolders).",
    )

    parser.add_argument(
        "--sampler",
        type=str,
        default="weighted",
        choices=["none", "weighted", "batch"],
        help="Sampling strategy for train loader: 'none' (shuffle), 'weighted' (per-image inverse freq), or 'batch' (stratified quotas per batch).",
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
        default=config.DEFAULT_BACKBONE_MODE,
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
        default=(config.DEFAULT_SCHEDULER or None),
        choices=["None", "ReduceLR", "OneCycle", "Cosine"],
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
        "--lambda_rect",
        type=float,
        default=config.LAMBDA_RECT,
        help="Weight for the rectangle loss (default: 0.1).",
    )
    parser.add_argument(
        "--lambda_child",
        type=float,
        default=config.LAMBDA_CHILD,
        help="Weight for the child face classification loss (default: 1).",
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
    parser.add_argument(
        "--sigma_l2_cls",
        type=float,
        default=config.SIGMA_L2_CLS,
        help="Sigma for L2 Loss, if used (default: None).",
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
        "--baby_thres",
        type=float,
        default=config.BABY_THRESH,
        help=f"Baby face confidence threshold for inference (default: {config.BABY_THRESH}).",
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

    parser.add_argument(
        "--bin_deg", type=int, default=10, help="Ancho de bin angular (grados)."
    )
    parser.add_argument(
        "--equalize_angle_bins",
        action="store_true",
        default=True,
        help="Forzar distribución uniforme/inverse-freq de ángulos en train.",
    )
    parser.add_argument(
        "--aug_bin_strategy",
        type=str,
        default="uniform",
        choices=["uniform", "inverse_freq"],
        help="Estrategia de muestreo de bins.",
    )
    parser.add_argument(
        "--max_rotate", type=float, default=180.0, help="Límite de rotación (grados)."
    )
    parser.add_argument(
        "--audit_aug_bins",
        action="store_true",
        default=False,
        help="Guardar histogramas pre y post augmentation en train.",
    )
    # Reproducibility
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    # Ablation mode
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Run an ablation over a list of lambda values, generating one run per value.",
    )
    parser.add_argument(
        "--ablate_param",
        type=str,
        default="lambda_rot",
        choices=["lambda_rot", "lambda_cls"],
        help="Which lambda to vary during ablation.",
    )
    parser.add_argument(
        "--ablate_values",
        type=float,
        nargs="+",
        default=None,
        help="List of values for the selected lambda (e.g. --ablate_values 2 4 8 16).",
    )
    parser.add_argument(
        "--ablate_seeds",
        type=int,
        nargs="+",
        default=None,
        help="Optional list of seeds to run for each lambda (e.g. --ablate_seeds 0 1 2).",
    )
    parser.add_argument(
        "--skip_inference",
        action="store_true",
        help="Skip inference stage after training (recommended for ablation).",
    )
    parser.add_argument(
        "--skip_compile",
        action="store_true",
        help="Disable torch.compile even on CUDA (recommended for repeated ablation runs).",
    )

    return parser.parse_args()


def build_run_suffix(ablate_param: str, ablate_value: float, seed: int) -> str:
    """
    Build a unique run suffix for ablation runs.

    Args:
        ablate_param: Parameter name being ablated (e.g., "lambda_rot").
        ablate_value: Value of the ablated parameter.
        seed: Random seed.

    Returns:
        A compact suffix string for naming runs and folders.
    """
    clean_param = ablate_param.replace("lambda_", "lam")
    return f"{clean_param}{ablate_value:g}_seed{seed}"


def run_single_experiment(args: argparse.Namespace) -> None:
    """
    Run a single training (and optional inference) experiment using the provided args.

    This function contains (almost) the same logic as your current main(), but assumes
    args.run_name and output directory are already set correctly for this run.

    Args:
        args: Parsed arguments for a single run.
    """
    print("[INFO] Starting training and inference with args:", vars(args))

    # ------------------------------------------------------------------------
    # 0. Reproducibility
    # ------------------------------------------------------------------------
    set_seed(args.seed)

    # ------------------------------------------------------------------------
    # I. Output directory structure
    # ------------------------------------------------------------------------
    output_dir = Path("runs") / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Output directory created at: {output_dir}")

    save_reproducibility_metadata(output_dir, vars(args))

    ckpt_path = output_dir / "checkpoint.pt"
    csv_path = output_dir / f"{args.run_name}.csv"
    config_path = output_dir / f"{args.run_name}.yaml"
    grids_dir = output_dir / "dataset_grids"
    anchor_preview_path = output_dir / "anchors_preview.jpg"
    inference_preview = output_dir / "training_grids"
    inference_preview.mkdir(exist_ok=True)
    grids_dir.mkdir(exist_ok=True)

    with open(config_path, "w") as f:
        yaml.dump(vars(args), f)
    print(f"[INFO] Saved config to {config_path}")

    # ------------------------------------------------------------------------
    # II. Setup
    # ------------------------------------------------------------------------
    device = get_default_device()
    print(f"[INFO] Using device: {device}")

    if args.backbone_mode == "train_all":
        norm_mean, norm_std = config.MEAN, config.STD
    else:
        norm_mean, norm_std = config.IMAGENET_MEAN, config.IMAGENET_STD

    img_size = tuple(args.img_size)

    labels_map = {
        0: "Leftside",
        1: "3/4 Leftside",
        2: "Frontal",
        3: "3/4 Rightside",
        4: "Rightside",
    }

    if args.audit_aug_bins:
        raw_train = BabyFacesDataset(args.root_dir, split="train", transform=None)
        pre_stats = collect_deg_by_class_from_dataset(raw_train, labels_map)
        angles_dir = output_dir / "angles"
        angles_dir.mkdir(exist_ok=True)
        plot_histograms_split(
            pre_stats, labels_map, args.bin_deg, angles_dir, tag="train_PRE"
        )
        bin_weights = build_bin_weights_from_degrees(pre_stats["all"], args.bin_deg)
    else:
        bin_weights = None

    train_transform = config.make_train_transform(
        img_size,
        args.use_augmentation,
        mean=norm_mean,
        std=norm_std,
        equalize=args.equalize_angle_bins,
        bin_deg=args.bin_deg,
        strategy=args.aug_bin_strategy,
        max_rotate=args.max_rotate,
        bin_weights=bin_weights,
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

    visualize_and_save_dataset_in_script(
        train_dataset,
        "train",
        grids_dir,
        num_images=args.batch_size,
        labels_map=labels_map,
    )
    visualize_and_save_dataset_in_script(
        val_dataset, "val", grids_dir, num_images=args.batch_size, labels_map=labels_map
    )

    if args.sampler == "weighted":
        sampler = make_weighted_sampler(train_dataset)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            collate_fn=custom_collate,
            num_workers=4,
            pin_memory=True,
            worker_init_fn=seed_worker,
        )
    elif args.sampler == "batch":
        batch_sampler, info = make_stratified_batch_sampler(
            train_dataset,
            batch_size=args.batch_size,
            seed=args.seed,
            replacement=True,
            drop_last=True,
        )
        print(f"[INFO] Stratified quotas (bs={args.batch_size}): {info['quota']}")
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=custom_collate,
            num_workers=4,
            pin_memory=True,
            worker_init_fn=seed_worker,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=custom_collate,
            num_workers=4,
            pin_memory=True,
            worker_init_fn=seed_worker,
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=custom_collate,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=seed_worker,
    )

    # ------------------------------------------------------------------------
    # IV. Model and loss setup
    # ------------------------------------------------------------------------
    model = NewBORN(args.backbone, args.out_channel, pretrained=args.use_pretrained).to(
        device
    )
    set_backbone_frozen(model, mode=args.backbone_mode)
    reset_heads(model)

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

    if args.resume_training:
        load_checkpoint_for_resuming(model, args.resume_training, device)

    if device.type == "cuda" and (not args.skip_compile):
        print("[INFO] Compiling model with torch.compile...")
        model = torch.compile(model)
        print("[INFO] Model compilation complete.")

    multitask_loss = MultiTaskLoss(
        args.obb_loss_type,
        args.rot_loss_type,
        args.cls_loss_type,
        args.lambda_cls,
        args.lambda_obb,
        args.lambda_rot,
        args.lambda_face,
        args.lambda_rect,
        args.lambda_child,
        args.pos_iou_thr_1,
        args.neg_iou_thr_1,
        args.pos_iou_thr_2,
        args.neg_iou_thr_2,
        args.alpha,
        args.gamma,
        args.neg_samples_ratio,
        args.face_pos_weight,
        args.sigma_l2_cls,
    )

    earlystopping = EarlyStopping(
        args.patience, verbose=True, delta=0.001, path=ckpt_path
    )

    if args.scheduler is None or args.scheduler == "None":
        args.scheduler = None

    # ------------------------------------------------------------------------
    # V. Training
    # ------------------------------------------------------------------------
    anchor_cache_path = config.ANCHORS_CACHE_PATH
    os.makedirs(os.path.dirname(anchor_cache_path), exist_ok=True)

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
        baby_thres=args.baby_thres,
        csv_path=csv_path,
        anchor_preview_path=anchor_preview_path,
        anchors_cache_path=anchor_cache_path,
        inference_preview=inference_preview,
    )

    print("\n[INFO] Training completed!")

    # ------------------------------------------------------------------------
    # VI. Inference (optional)
    # ------------------------------------------------------------------------
    if args.skip_inference:
        print("[INFO] Skipping inference stage (--skip_inference).")
        return

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
        worker_init_fn=seed_worker,
    )

    visualize_and_save_dataset_in_script(
        test_dataset, "test", grids_dir, num_images=9, labels_map=labels_map
    )

    trained_model = NewBORN(
        args.backbone, args.out_channel, pretrained=args.use_pretrained
    ).to(device)

    print(f"[INFO] Loading weights from: {ckpt_path}")
    raw_weights = torch.load(ckpt_path, map_location=device)
    state_dict = raw_weights.get("model_state_dict", raw_weights)

    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        stripped = {}
        for k, v in state_dict.items():
            if k.startswith("_orig_mod."):
                stripped[k[len("_orig_mod.") :]] = v
            else:
                stripped[k] = v
        state_dict = stripped

    trained_model.load_state_dict(state_dict)
    trained_model.eval()

    inference(
        trained_model,
        test_loader=test_loader,
        output_dir=output_dir,
        device=device,
        labels_map=labels_map,
        scale_factors=config.SCALE_FACTORS,
        ratio_factors=config.RATIO_FACTORS,
        face_thres=args.face_thres,
        iou_thres=args.iou_thres,
        class_thres=args.class_thres,
        baby_thres=args.baby_thres,
        grid_shape=(args.grid_rows, args.grid_cols),
        mean=norm_mean,
        std=norm_std,
        anchors_cache_path=anchor_cache_path,
    )

    print(f"[INFO] Plotting training curves from {csv_path}")
    plot_training_curves_from_csv(csv_path, output_dir)
    print(f"[INFO] All done! Check {output_dir} for results.")


def main() -> None:
    """
    Entry point. Runs either a single experiment or an ablation over lambda values.
    """
    args = parse_args()

    if args.ablation:
        if not args.ablate_values:
            raise ValueError("Ablation enabled but --ablate_values was not provided.")

        seeds = args.ablate_seeds if args.ablate_seeds else [args.seed]

        base_run_name = args.run_name
        print(
            f"[INFO] Running ablation: param={args.ablate_param}, values={args.ablate_values}, seeds={seeds}"
        )

        for ablate_value in args.ablate_values:
            for seed in seeds:
                # Create a copy-like behavior by updating args in place, then restoring.
                args.seed = seed

                if args.ablate_param == "lambda_rot":
                    args.lambda_rot = float(ablate_value)
                elif args.ablate_param == "lambda_cls":
                    args.lambda_cls = float(ablate_value)
                else:
                    raise ValueError(f"Unknown ablation parameter: {args.ablate_param}")

                suffix = build_run_suffix(args.ablate_param, float(ablate_value), seed)
                args.run_name = f"{base_run_name}_{suffix}"

                # Strong recommendation for ablation
                args.skip_inference = True

                # Run
                run_single_experiment(args)

                # Optional: free CUDA memory between runs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        print("[INFO] Ablation completed.")
        return

    # Single run
    run_single_experiment(args)


if __name__ == "__main__":
    main()
