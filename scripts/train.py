from typing import Optional

from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.data import build_detection_train_loader
from detectron2.config import get_cfg

from data_setup import dataset_mapper


class FaceTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(
        cls,
        cfg: get_cfg().CONFIG_CLASS,
        dataset_name: str,
        output_folder: Optional[str] = None,
    ):
        """
        Creates an evaluator for the specified dataset.

        Args:
            cfg (get_cfg().CONFIG_CLASS): Configuration for the model and training.
            dataset_name (str): Name of the dataset for evaluation.
            output_folder (Optional[str]): Output folder to store evaluation results.

        Returns:
            COCOEvaluator: A COCO evaluator for the dataset.
        """
        # Return a COCO evaluator for the specified dataset
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    @classmethod
    def build_train_loader(cls, cfg: get_cfg().CONFIG_CLASS):
        """
        Creates a data loader for training.

        Args:
            cfg (get_cfg().CONFIG_CLASS): Configuration for the model and training.

        Returns:
            DataLoader: A data loader for training.
        """
        # Return a custom data loader for training, using 'dataset_mapper'
        return build_detection_train_loader(cfg, mapper=dataset_mapper)
