from detectron2.config import get_cfg
from detectron2 import model_zoo

def setup_cfg(base_config_path: str, rotated_bbox_config_path: str, 
            pretrained_model_url: str, train_dataset: str, val_dataset: str, 
            output_dir: str, num_classes: int, num_workers: int, 
            device: str, pretrained_weights_url: str, freeze_backbone: bool, 
            freeze_at_block: int, ims_per_batch: int, checkpoint_period: int, 
            base_lr: float, max_iter: int, batch_size_per_image: int, 
            lr_decay_steps: tuple) -> get_cfg().CONFIG_CLASS:
    """
    Sets up and returns a Detectron2 configuration for training a model with custom settings.

    Args:
        base_config_path (str): Path to the base configuration file.
        rotated_bbox_config_path (str): Path to the configuration file for rotated bounding boxes.
        pretrained_model_url (str): URL to get the configuration file of the pretrained model.
        train_dataset (str): Name of the training dataset.
        val_dataset (str): Name of the validation dataset.
        output_dir (str): Directory for output data (models, logs).
        num_classes (int): Number of classes for ROI heads.
        num_workers (int): Number of data loading workers.
        device (str): Model device type ('cuda' for GPU or 'cpu').
        pretrained_weights_url (str): URL to the pre-trained weights.
        freeze_backbone (bool): Flag to freeze the backbone layers.
        freeze_at_block (int): The stage in the backbone at which to stop freezing (only if freeze_backbone is True).
        ims_per_batch (int): Number of images per batch during training.
        checkpoint_period (int): Period to save the model checkpoint.
        base_lr (float): Base learning rate.
        max_iter (int): Maximum number of iterations for training.
        batch_size_per_image (int): Batch size per image during ROI head computation.
        lr_decay_steps (tuple): Steps for learning rate decay.

    Returns:
        get_cfg().CONFIG_CLASS: Configured Detectron2 configuration object for training.
    """
    cfg = get_cfg()
    # Load the base model configuration
    cfg.merge_from_file(model_zoo.get_config_file(base_config_path))

    # Load custom configuration for rotated bounding boxes
    cfg.merge_from_file(rotated_bbox_config_path)

    # Set model weights from a pre-trained model
    cfg.MODEL.WEIGHTS = pretrained_model_url

    # Set the training and testing datasets
    cfg.DATASETS.TRAIN = (train_dataset,)
    cfg.DATASETS.TEST = (val_dataset,)

    # Set the output directory for training artifacts
    cfg.OUTPUT_DIR = output_dir

    # Set the number of classes for ROI heads
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes

    # Set the number of data loading workers
    cfg.DATALOADER.NUM_WORKERS = num_workers

    # Set the device type for training (GPU or CPU)
    cfg.MODEL.DEVICE = device

    # Load pre-trained weights for the model
    cfg.MODEL.WEIGHTS = pretrained_weights_url

    # Conditionally freeze the backbone at the specified block
    if freeze_backbone:
        cfg.MODEL.BACKBONE.FREEZE_AT = freeze_at_block

    # Set other training parameters
    cfg.SOLVER.IMS_PER_BATCH = ims_per_batch
    cfg.SOLVER.CHECKPOINT_PERIOD = checkpoint_period
    cfg.SOLVER.BASE_LR = base_lr
    cfg.SOLVER.MAX_ITER = max_iter
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = batch_size_per_image
    cfg.SOLVER.STEPS = lr_decay_steps

    # Additional configurations can be set here as needed

    return cfg