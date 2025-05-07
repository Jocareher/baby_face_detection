import math

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

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# =======================
# Default Hyperparameters
# =======================
DEFAULT_EPOCHS = 100
DEFAULT_LR = 1e-4
DEFAULT_BATCH_SIZE = 32
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_OPTIMIZER = "ADAM"
DEFAULT_SCHEDULER = None
DEFAULT_CLIP_VALUE = None
DEFAULT_GRAD_CLIP_MODE = "Norm"
DEFAULT_PATIENCE = 10

# =======================
# Precomputed OBB Statistics
# =======================
PRECOMPUTED_OBB_STATS = {
    (640, 640): {"avg_size": 209.56, "avg_ratio": 1.1851},
    (512, 512): {"avg_size": 167.65, "avg_ratio": 1.1851},
    (256, 256): {"avg_size": 83.83, "avg_ratio": 1.1851},
    (224, 224): {"avg_size": 73.35, "avg_ratio": 1.1851},
}

# =======================
# Anchor Generation Params
# =======================
SCALE_FACTORS = [0.75, 1.0, 1.2]
RATIO_FACTORS = [1.1851]
ANGLES = [
    -math.pi / 2,
    -math.pi / 3,
    -math.pi / 6,
    0.0,
    math.pi / 6,
    math.pi / 3,
    math.pi / 2,
]
NUM_ANCHORS = len(SCALE_FACTORS) * len(RATIO_FACTORS) * len(ANGLES)


# =======================
# Loss Function Weights
# =======================
ALPHA = [1.5, 1.5, 1.5, 2.0, 2.0, 0.5]
GAMMA = 2.0
POS_IOU_THRESH = 0.2
LAMBDA_CLS = 1.0
LAMBDA_OBB = 1.0
LAMBDA_ROT = 1.0


# =======================
# WandB Configuration
# =======================
PROJECT_NAME = "RetinaBabyFace"
RUN_NAME = "run_1"


# =======================
# Data Transforms
# =======================
def get_train_transform(
    img_size=(640, 640), use_augmentation=True, mean=IMAGENET_MEAN, std=IMAGENET_STD
):
    """
    Returns a composition of training transforms. Normalization stats can be overridden.
    """
    norm = ToTensorNormalize(mean=mean, std=std)
    if use_augmentation:
        return transforms.Compose(
            [
                RandomHorizontalFlipOBB(prob=0.5),
                RandomRotateOBB(max_angle=30, prob=0.3),
                ColorJitterOBB(brightness=0.2, contrast=0.2, saturation=0.2, prob=0.5),
                RandomNoiseOBB(std=10, prob=0.5),
                RandomBlurOBB(ksize=(5, 5), prob=0.3),
                RandomOcclusionOBB(max_size_ratio=0.3, prob=0.3),
                Resize(img_size),
                norm,
            ]
        )
    else:
        return transforms.Compose(
            [
                Resize(img_size),
                norm,
            ]
        )


def get_val_transform(img_size=(640, 640), mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """
    Returns a composition of validation transforms. Normalization stats can be overridden.
    """
    return transforms.Compose(
        [
            Resize(img_size),
            ToTensorNormalize(mean=mean, std=std),
        ]
    )


# RandomScaleTranslateOBB(
#     scale_range=(0.8, 1.1), translate_range=(-0.2, 0.2), prob=0.3
# ),
