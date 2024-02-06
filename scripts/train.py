from typing import Optional

from detectron2.engine import DefaultTrainer
from detectron2.evaluation import RotatedCOCOEvaluator, DatasetEvaluators
from detectron2.data import build_detection_train_loader
from detectron2.config import CfgNode
from detectron2.engine.hooks import HookBase
from detectron2.evaluation import inference_context
from detectron2.utils.logger import log_every_n_seconds
from detectron2.data import DatasetMapper, build_detection_test_loader
import detectron2.utils.comm as comm
import torch
import time
import datetime
import numpy as np

from data_setup import dataset_mapper

class LossEvalHook(HookBase):
    def __init__(self, eval_period, model, data_loader):
        self._model = model
        self._period = eval_period
        self._data_loader = data_loader
    
    def _do_loss_eval(self):
        # Copying inference_on_dataset from evaluator.py
        total = len(self._data_loader)
        num_warmup = min(5, total - 1)
            
        start_time = time.perf_counter()
        total_compute_time = 0
        losses = []
        for idx, inputs in enumerate(self._data_loader):            
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0
            start_compute_time = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Loss on Validation  done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    ),
                    n=5,
                )
            loss_batch = self._get_loss(inputs)
            losses.append(loss_batch)
        mean_loss = np.mean(losses)
        self.trainer.storage.put_scalar('validation_loss', mean_loss)
        comm.synchronize()

        return losses
            
    def _get_loss(self, data):
        # How loss is calculated on train_loop 
        metrics_dict = self._model(data)
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        total_losses_reduced = sum(loss for loss in metrics_dict.values())
        return total_losses_reduced
        
        
    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self._do_loss_eval()
        self.trainer.storage.put_scalars(timetest=12)

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
    
    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1,LossEvalHook(
            self.cfg.TEST.EVAL_PERIOD,
            self.model,
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                DatasetMapper(self.cfg,True)
            )
        ))
        return hooks
