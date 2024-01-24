import os
import time as t

from detectron2.data import build_detection_test_loader
from detectron2.evaluation import inference_on_dataset

import data_setup, engine, utils, parser, train


def main_worker():  # local_rank, config
    # Setting random seed for reproducibility
    utils.set_seed()

    # Parsing command line arguments for model configuration
    config = parser.parse_arguments()

    # Set the root directory
    root_dir = config["DATASET"]["root_dir"]

    # Define the class labels
    class_labels = config["DATASET"]["class_labels"]

    # Register the dataset for training and get datasets sizes
    train_size, val_size, test_size = data_setup.register_datasets(
        root_dir=root_dir, class_labels=class_labels
    )

    # DATALOADER config
    train_dataset = config["DATALOADER"]["train_dataset"]
    val_dataset = config["DATALOADER"]["val_dataset"]
    num_workers = config["DATALOADER"]["num_workers"]

    # Model config
    base_config_path = config["MODEL"]["base_config_path"]
    rotated_bbox_config_path = config["MODEL"]["rotated_bbox_config_path"]
    pretrained_model_url = config["MODEL"]["pretrained_model_url"]
    num_classes = config["MODEL"]["num_classes"]
    device = config["MODEL"]["device"]
    unfreeze_backbone = config["MODEL"]["unfreeze_backbone"]
    freeze_at_block = config["MODEL"]["freeze_at_block"]
    output_dir = config["MODEL"]["output_dir"]

    # Training config
    ims_per_batch = config["TRAINING"]["ims_per_batch"]
    checkpoint_period = config["TRAINING"]["checkpoint_period"]
    base_lr = config["TRAINING"]["base_lr"]
    epochs = config["TRAINING"]["epochs"]
    batch_size_per_image = config["TRAINING"]["batch_size_per_image"]
    warm_steps = config["TRAINING"]["warm_steps"]
    gamma = config["TRAINING"]["gamma"]

    # Set the model and training configuration
    model_and_train_config = engine.setup_cfg(
        base_config_path=base_config_path,
        rotated_bbox_config_path=rotated_bbox_config_path,
        pretrained_model_url=pretrained_model_url,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=output_dir,
        num_classes=num_classes,
        num_workers=num_workers,
        device=device,
        freeze_backbone=unfreeze_backbone,
        freeze_at_block=freeze_at_block,
        ims_per_batch=ims_per_batch,
        checkpoint_period=checkpoint_period,
        base_lr=base_lr,
        epochs=epochs,
        train_data_size=train_size,
        batch_size_per_image=batch_size_per_image,
        warm_steps=warm_steps,
        gamma=gamma,
    )

    # if local_rank == 0:
    #     # Only main process write the outputs and logs
    os.makedirs(model_and_train_config.OUTPUT_DIR, exist_ok=True)

    # Check model defined config and output training mode
    freeze_backbone_message = "The model is using pre-trained weights" if not unfreeze_backbone else f"The model is in fine-tuning mode, keeping frozen the weights till the {freeze_at_block} conv block"
    print(freeze_backbone_message)
    
    # Print output directory
    print(f"Output Directory: {output_dir}")

    # Print other training configurations
    print(
        f"\nTraining with the following configuration: "
        f"\nBatch size: {model_and_train_config.SOLVER.IMS_PER_BATCH}"
        f"\nCheckpoint period: {model_and_train_config.SOLVER.CHECKPOINT_PERIOD}"
        f"\nBase learning rate: {model_and_train_config.SOLVER.BASE_LR}"
        f"\nMax iterations: {model_and_train_config.SOLVER.MAX_ITER}"
        f"\nBatch size per ROI heads: {model_and_train_config.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE}"
        f"\nSolver steps: {model_and_train_config.SOLVER.STEPS}"
        f"\nWarm-up iterations: {model_and_train_config.SOLVER.WARMUP_ITERS}"
        f"\nGamma: {model_and_train_config.SOLVER.GAMMA}"
    )


    # Initialize the trainer
    trainer = train.FaceTrainer(model_and_train_config)
    trainer.resume_or_load(resume=False)

    # Measure training time
    start_time = t.time()
    trainer.train()
    training_time = t.time() - start_time
    print(f"Training time: {training_time} seconds")

    # Evaluate the performance after the training
    # if local_rank == 0:  # Evaluate only on the main process
    evaluator = train.FaceTrainer.build_evaluator(
        model_and_train_config, model_and_train_config.DATASETS.TEST
    )
    val_loader = build_detection_test_loader(
        model_and_train_config, model_and_train_config.DATASETS.TEST
    )
    metrics = inference_on_dataset(trainer.model, val_loader, evaluator)
    print(metrics)


# def main():
#     # Parsing command line arguments for model configuration
#     config = parser.parse_arguments()

#     # Number of GPUs to use (set this to the number of available GPUs)
#     num_gpus = int(config["DATALOADER"]["num_gpus"])

#     print("Config before launch:", config)

#     try:
#         if torch.cuda.is_available():
#             num_gpus = torch.cuda.device_count()
#     except ImportError:
#         print("Assuming a single gpu")


#     launch(
#         main_worker,
#         num_gpus,
#         args=(config,),
#     )

if __name__ == "__main__":
    main_worker()
