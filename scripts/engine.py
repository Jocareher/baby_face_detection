from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.config import CfgNode


def setup_cfg(
    base_config_path: str,
    rotated_bbox_config_path: str,
    pretrained_model_url: str,
    train_dataset: str,
    val_dataset: str,
    output_dir: str,
    num_classes: int,
    num_workers: int,
    device: str,
    freeze_backbone: bool,
    freeze_at_block: int,
    ims_per_batch: int,
    checkpoint_period: int,
    base_lr: float,
    epochs: int,
    train_data_size: int,
    batch_size_per_image: int,
    warm_steps: int,
    gamma: float,
) -> CfgNode:
    """
    Sets up and returns a Detectron2 configuration for training a model with custom settings.

    Args:
        - base_config_path (str): Path to the backbone configuration file.
        - rotated_bbox_config_path (str): Path to the configuration file for rotated bounding boxes.
        - pretrained_model_url (str): URL to get the configuration file of the pretrained model.
        - train_dataset (str): Name of the training dataset.
        - val_dataset (str): Name of the validation dataset.
        - output_dir (str): Directory for output data (models, logs).
        - num_classes (int): Number of classes for ROI heads.
        - num_workers (int): Number of data loading workers.
        - device (str): Model device type ('cuda' for GPU or 'cpu').
        - freeze_backbone (bool): Flag to freeze the backbone layers.
        - freeze_at_block (int): The stage in the backbone at which to stop freezing (only if freeze_backbone is True).
        - ims_per_batch (int): Number of images per batch during training.
        - checkpoint_period (int): Period to save the model checkpoint.
        - base_lr (float): Base learning rate.
        - epochs (int): Number of training epochs.
        - train_data_size (int): Size of the training dataset.
        - batch_size_per_image (int): Batch size per image during ROI head computation.
        - warm_steps (int): Number of iterations for the warmup phase.
        - gamma (float): Factor for learning rate decay.

    Returns:
        CfgNode: Configured Detectron2 configuration object for training.
    """
    cfg = get_cfg()
    # Load the base configuration from a pre-defined model in the model zoo
    cfg.merge_from_file(model_zoo.get_config_file(base_config_path))

    # Set URL for the pre-trained model's configuration
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(pretrained_model_url)

    # Incorporate custom configurations for handling rotated bounding boxes
    cfg.merge_from_file(rotated_bbox_config_path)

    # Specify the names of the training and validation datasets
    cfg.DATASETS.TRAIN = (train_dataset,)
    cfg.DATASETS.TEST = (val_dataset,)

    # Define the directory where model outputs and logs will be saved
    cfg.OUTPUT_DIR = output_dir

    # Set the number of object classes for Region of Interest (ROI) heads
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes

    # Set the number of workers for data loading processes
    cfg.DATALOADER.NUM_WORKERS = num_workers

    # Choose the computation device for model training (CPU or GPU)
    cfg.MODEL.DEVICE = device

    # Optionally freeze the early layers of the model's backbone
    if freeze_backbone:
        cfg.MODEL.BACKBONE.FREEZE_AT = freeze_at_block

    # Training parameters
    cfg.SOLVER.IMS_PER_BATCH = ims_per_batch  # Number of images per batch
    cfg.SOLVER.CHECKPOINT_PERIOD = checkpoint_period  # Frequency of checkpoint saving
    cfg.SOLVER.BASE_LR = base_lr  # Starting learning rate
    cfg.SOLVER.MAX_ITER = epochs * (
        train_data_size // ims_per_batch
    )  # Maximum iterations for training
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
        batch_size_per_image  # Batch size per image in ROI heads
    )
    cfg.SOLVER.STEPS = (
        int(cfg.SOLVER.MAX_ITER * 0.6),
        int(cfg.SOLVER.MAX_ITER * 0.8),
    )  # Iterations at which to decay learning rate
    cfg.SOLVER.WARMUP_ITERS = (
        warm_steps  # Iterations for warmup phase at start of training
    )
    cfg.SOLVER.GAMMA = gamma  # Learning rate decay factor

    # Additional configurations can be set here as needed

    return cfg
