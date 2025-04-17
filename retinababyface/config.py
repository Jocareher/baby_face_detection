from torchvision import transforms
from data_setup.augmentations import (
    RandomHorizontalFlipOBB,
    RandomRotateOBB,
    RandomScaleTranslateOBB,
    ColorJitterOBB,
    RandomNoiseOBB,
    RandomBlurOBB,
    RandomOcclusionOBB,
    Resize,
    ToTensorNormalize,
)

# =======================
# Image Normalization
# =======================
MEAN = (0.6427, 0.5918, 0.5525)
STD = (0.2812, 0.2825, 0.3036)

# =======================
# Default Hyperparameters
# =======================
DEFAULT_EPOCHS = 50
DEFAULT_LR = 1e-4
DEFAULT_BATCH_SIZE = 4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_OPTIMIZER = "ADAM"
DEFAULT_SCHEDULER = None
DEFAULT_CLIP_VALUE = None
DEFAULT_GRAD_CLIP_MODE = "Norm"
DEFAULT_PATIENCE = 5

# =======================
# Anchor Generation Params
# =======================
BASE_SIZE = 209.56
BASE_RATIO = 1.1851
SCALE_FACTORS = [0.75, 1.0, 1.25]
RATIO_FACTORS = [0.85, 1.0, 1.15]

# =======================
# WandB Configuration
# =======================
PROJECT_NAME = "RetinaBabyFace"
RUN_NAME = "run_1"


# =======================
# Data Transforms
# =======================
def get_train_transform():
    return transforms.Compose(
        [
            RandomHorizontalFlipOBB(prob=0.5),
            RandomRotateOBB(max_angle=30, prob=0.3),
            RandomScaleTranslateOBB(
                scale_range=(0.8, 1.1), translate_range=(-0.2, 0.2), prob=0.3
            ),
            ColorJitterOBB(brightness=0.2, contrast=0.2, saturation=0.2, prob=0.5),
            RandomNoiseOBB(std=10, prob=0.5),
            RandomBlurOBB(ksize=(5, 5), prob=0.3),
            RandomOcclusionOBB(max_size_ratio=0.3, prob=0.3),
            Resize((640, 640)),
            ToTensorNormalize(mean=MEAN, std=STD),
        ]
    )


def get_val_transform():
    return transforms.Compose(
        [
            Resize((640, 640)),
            ToTensorNormalize(mean=MEAN, std=STD),
        ]
    )
