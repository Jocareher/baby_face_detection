from typing import Optional

from detectron2.engine import DefaultTrainer
from detectron2.evaluation import RotatedCOCOEvaluator, DatasetEvaluators
from detectron2.data import build_detection_train_loader
from detectron2.config import CfgNode

from data_setup import dataset_mapper


class FaceTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(
        cls,
        cfg: CfgNode,
        dataset_name: str,
        output_folder: Optional[str] = None,
    ):
        """
        Instantiate an evaluator for rotated bounding box detection on a given dataset.
        This method overrides the DefaultTrainer's build_evaluator method to provide
        a COCO-style evaluator that supports the evaluation of rotated bounding boxes.

        Args:
            cfg (CfgNode): Configuration Node object containing the model and training configurations.
            dataset_name (str): The name of the dataset to be evaluated.
            output_folder (Optional[str], optional): The directory where the evaluation results will be stored.
                If not provided, the results won't be saved to disk.

        Returns:
            DatasetEvaluators: An object containing a list of evaluators, with a RotatedCOCOEvaluator
            for evaluating datasets with rotated bounding boxes in COCO format. This object can be
            used to evaluate the dataset during training or inference.
        """
        # Instantiate the RotatedCOCOEvaluator
        evaluator = RotatedCOCOEvaluator(dataset_name, cfg, True, output_folder)

        # Return a DatasetEvaluators object containing the list of evaluators
        return DatasetEvaluators([evaluator])

    @classmethod
    def build_train_loader(cls, cfg: CfgNode):
        """
        Creates a data loader for training.

        Args:
            cfg (get_cfg().CONFIG_CLASS): Configuration for the model and training.

        Returns:
            DataLoader: A data loader for training.
        """
        # Return a custom data loader for training, using 'dataset_mapper'
        return build_detection_train_loader(cfg, mapper=dataset_mapper)
