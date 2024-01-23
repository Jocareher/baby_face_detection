import yaml
import argparse
from typing import Union
import os


def parse_arguments():
    """
    Parses command line arguments provided by the user, updating the model configuration accordingly,
    and returns the updated configuration.
    """
    
    # Load and update configuration
    config_file = "./configs/config_train.yaml"
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    parser = argparse.ArgumentParser(
        description="Update Detectron2 training configuration."
    )

    # Define command line arguments
    parser.add_argument(
        "--root_dir",
        default="data/face_dataset",
        help="Root directory for the datasets.",
    )
    parser.add_argument(
        "--batch_size", default=4, type=int, help="Batch size for training."
    )
    parser.add_argument(
        "--num_workers", default=4, type=int, help="Number of data loading workers."
    )
    parser.add_argument("--device", default="cuda", type=str, help="Computation device for model training")
    parser.add_argument(
        "--base_config_path",
        default="COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
        help="Base configuration path.",
    )
    parser.add_argument(
        "--pretrained_model_url",
        default="COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
        help="URL for the pretrained model.",
    )
    parser.add_argument(
        "--freeze_backbone",
        default=False,
        type=bool,
        help="Whether to freeze the backbone.",
    )
    parser.add_argument(
        "--freeze_at_block",
        default=3,
        type=int,
        help="Block at which to stop freezing the backbone.",
    )
    parser.add_argument(
        "--ims_per_batch", default=4, type=int, help="Images per batch."
    )
    parser.add_argument(
        "--checkpoint_period",
        default=1000,
        type=int,
        help="Period to save checkpoints.",
    )
    parser.add_argument(
        "--base_lr", default=0.0025, type=float, help="Base learning rate."
    )
    parser.add_argument(
        "--epochs", default=100, type=int, help="Number of training epochs."
    )
    parser.add_argument(
        "--batch_size_per_image",
        default=128,
        type=int,
        help="Batch size per image for ROI heads.",
    )
    parser.add_argument(
        "--warm_steps", default=1000, type=int, help="Number of warm-up steps."
    )
    parser.add_argument(
        "--gamma", default=0.1, type=float, help="Gamma value for learning rate decay."
    )

    args = parser.parse_args()

    # Update config with provided arguments
    config["DATASET"]["root_dir"] = args.root_dir
    config["DATALOADER"]["num_workers"] = args.num_workers
    config["MODEL"]["base_config_path"] = args.base_config_path
    config["MODEL"]["rotated_bbox_config_path"] = os.path.join(
        os.path.dirname(config_file), "rotated_bbox_config.yaml"
    )
    config["MODEL"]["device"] = args.device
    config["MODEL"]["pretrained_model_url"] = args.pretrained_model_url
    config["MODEL"]["freeze_backbone"] = args.freeze_backbone
    config["MODEL"]["freeze_at_block"] = args.freeze_at_block
    config["TRAINING"]["ims_per_batch"] = args.ims_per_batch
    config["TRAINING"]["checkpoint_period"] = args.checkpoint_period
    config["TRAINING"]["base_lr"] = args.base_lr
    config["TRAINING"]["epochs"] = args.epochs
    config["TRAINING"]["batch_size_per_image"] = args.batch_size_per_image
    config["TRAINING"]["warm_steps"] = args.warm_steps
    config["TRAINING"]["gamma"] = args.gamma

    # Save updated configuration
    with open(config_file, "w") as file:
        yaml.safe_dump(config, file)

    return config


if __name__ == "__main__":
    updated_config = parse_arguments()
    print("Updated configuration:", updated_config)