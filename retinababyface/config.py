import math

from torchvision import transforms
from data_setup.augmentations import (
    RandomHorizontalFlipOBB,
    RandomRotateOBB,
    RandomScaleTranslateOBB,
    RandomGrayOBB,
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
DEFAULT_PATIENCE = 3
DEFAULT_OUT_CHANNELS = 64
DEFAULT_BACKBONE_MODE = "feature_extractor"

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
# SCALE_FACTORS = [
#     0.19,
#     0.37,
#     0.59,
#     0.70,
# ]  # Values obtained from k-means clustering on the training set
# RATIO_FACTORS = [
#     0.7,
#     1.08,
#     1.15,
# ]  # Values obtained from k-means clustering on the training set
# ANGLES = [
#     -1.4548,  # ≈ -83.35°
#     -0.6445,  # ≈ -36.93°
#     -0.1704,  # ≈ -9.76°
#     0.0830,  # ≈  4.75°
#     0.5354,  # ≈ 30.68°
#     1.4058,  # ≈ 80.55°
#     -0.0706,  # ≈ 175.93° → –0.07 rad
# ]  # Values obtained from k-means clustering on the training set
# The following values are used in the RetinaFace paper
# and are not based on k-means clustering.
# They are included for reference and can be used if desired.

BASE_ANCHOR_SIZES = [
    16.0,  # P2  (stride=4)
    32.0,  # P3  (stride=8)
    64.0,  # P4  (stride=16)
    128.0,  # P5  (stride=32)
    256.0,  # P6  (stride=64)
]  # Base anchor sizes for different feature map levels (P2 to P6)

SCALE_FACTORS = [
    2 ** (0 / 3),  # = 1.0
    2 ** (1 / 3),  # ≈ 1.2599
    2 ** (2 / 3),  # ≈ 1.5874
]  # Values from RetinaFace paper
RATIO_FACTORS = [1.0]
ANGLES = [0.0]
NUM_ANCHORS = len(SCALE_FACTORS) * len(RATIO_FACTORS) * len(ANGLES)


# =======================
# Loss Function Weights
# =======================
ALPHA = [
    1.0,
    1.0,
    0.56,
    3.33,
    3.33,
]  # Values according to dataset's distribution (old_values = [1.5, 1.5, 1.5, 2.5, 2.5])
GAMMA = 2.0
POS_IOU_THRESH_1 = 0.6
NEG_IOU_THRESH_1 = 0.3
POS_IOU_THRESH_2 = 0.5
NEG_IOU_THRESH_2 = 0.4
NEG_SAMPLES_RATIO = 5
LAMBDA_CLS = 1.0
LAMBDA_FACE = 1.0
LAMBDA_OBB = 1.0
LAMBDA_ROT = 1.0
OBB_LOSS_TYPE = "smooth_l1"  # "smooth_l1", "l1"

# =======================
# Inference Parameters
# =======================
CONF_THRESH = 0.5
IOU_THRESH = 0.3
CLASS_THRESH = 0.6

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
                # RandomHorizontalFlipOBB(prob=0.5),
                RandomRotateOBB(max_angle=30, prob=0.3),
                # RandomScaleTranslateOBB(
                #     scale_range=(0.85, 1.15),
                #     translate_range=(-0.1, 0.1),
                #     prob=0.3,
                # ),
                RandomOcclusionOBB(max_size_ratio=0.5, prob=0.5),
                RandomNoiseOBB(std=10, prob=0.7),
                RandomBlurOBB(ksize=(5, 5), prob=0.7),
                RandomGrayOBB(prob=0.3),
                ColorJitterOBB(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    prob=0.7,
                    hue=0.05,
                    gamma=0.1,
                ),
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
