# ==============================================================================
#
# PlantXAI-Net
#
# Module 2 : EfficientNet-B0 Training Pipeline
#
# Phase 1 : Project Configuration
#
# Part 1 : Project Information + Imports + Global Constants
#
# Version : 1.0
#
# Author : Gursimran Singh
#
# Description :
# This module trains an EfficientNet-B0 model on the balanced PlantVillage
# dataset using Transfer Learning. The training pipeline includes:
#
# • EfficientNet-B0 (ImageNet Pretrained)
# • Mixed Precision Training (AMP)
# • Early Stopping
# • Learning Rate Scheduler
# • Automatic Checkpoint Saving
# • Resume Training
# • Performance Evaluation
# • Confusion Matrix
# • Classification Report
# • Training Curves
#
# This module is designed for PlantXAI-Net and follows research-grade coding
# practices suitable for IEEE/Scopus publications.
#
# ==============================================================================


# ==============================================================================
# Standard Library Imports
# ==============================================================================

import os
import time
import random
import logging
import warnings

from pathlib import Path
from typing import Dict
from typing import List
from typing import Tuple
from typing import Optional


# ==============================================================================
# Numerical Computing
# ==============================================================================

import numpy as np
import pandas as pd


# ==============================================================================
# Deep Learning
# ==============================================================================

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from torchvision import datasets
from torchvision import transforms
from torchvision import models


# ==============================================================================
# Evaluation Metrics
# ==============================================================================

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# ==============================================================================
# Visualization
# ==============================================================================

import matplotlib.pyplot as plt


# ==============================================================================
# Progress Bar
# ==============================================================================

from tqdm import tqdm


# ==============================================================================
# Ignore Warnings
# ==============================================================================

warnings.filterwarnings("ignore")


# ==============================================================================
# Global Project Information
# ==============================================================================

PROJECT_NAME = "PlantXAI-Net"

MODULE_NAME = "EfficientNet-B0 Training Pipeline"

MODULE_VERSION = "1.0"

MODEL_NAME = "EfficientNet-B0"

FRAMEWORK = "PyTorch"

AUTHOR = "Gursimran Singh"



# ==============================================================================
# End of Part 1
# ==============================================================================

# ==============================================================================
#
# Phase 1
#
# Part 2
#
# Directory Configuration + Global Configuration
#
# ==============================================================================


# ==============================================================================
# Directory Configuration
# ==============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_ROOT = (
    PROJECT_ROOT
    / "dataset"
    / "PlantVillage"
)

TRAIN_DIR = (
    DATASET_ROOT
    / "balanced"
)

VALIDATION_DIR = (
    DATASET_ROOT
    / "split"
    / "validation"
)

TEST_DIR = (
    DATASET_ROOT
    / "split"
    / "test"
)

MODEL_ROOT = (
    PROJECT_ROOT
    / "models"
)

CHECKPOINT_DIR = (
    MODEL_ROOT
    / "checkpoints"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)

LOG_DIR = (
    RESULTS_DIR
    / "logs"
)

PLOT_DIR = (
    RESULTS_DIR
    / "plots"
)

METRIC_DIR = (
    RESULTS_DIR
    / "metrics"
)

REPORT_DIR = (
    RESULTS_DIR
    / "reports"
)


# ==============================================================================
# Training Hyperparameters
# ==============================================================================

IMAGE_SIZE = 224

BATCH_SIZE = 32

NUM_EPOCHS = 5

LEARNING_RATE = 0.001

WEIGHT_DECAY = 1e-4

NUM_WORKERS = 0

PIN_MEMORY = True

SHUFFLE = True

RANDOM_SEED = 42

EARLY_STOPPING_PATIENCE = 7

SAVE_BEST_MODEL = True

SAVE_CHECKPOINT = True


# ==============================================================================
# Device Configuration
# ==============================================================================

DEVICE = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else

    "cpu"

)

USE_AMP = torch.cuda.is_available()


# ==============================================================================
# Dataset Information
#
# NOTE:
# These values will be initialized after the ImageFolder datasets
# are created in Phase 2.
# ==============================================================================

CLASS_NAMES = []

CLASS_TO_INDEX = {}

INDEX_TO_CLASS = {}

NUM_CLASSES = 0


# ==============================================================================
# End of Part 2
# ==============================================================================

# ==============================================================================
# Set Random Seed
# ==============================================================================

def set_random_seed(
    seed: int = RANDOM_SEED
) -> None:
    """
    Set random seed for reproducibility.

    Parameters
    ----------
    seed : int
        Random seed value.

    Returns
    -------
    None
    """

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


# ==============================================================================
# Create Project Directories
# ==============================================================================

def create_directories() -> None:
    """
    Create all required project directories.

    Returns
    -------
    None
    """

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    PLOT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    METRIC_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


# ==============================================================================
# Configure Logging
# ==============================================================================

LOG_FILE = (
    LOG_DIR
    / "training_pipeline.log"
)


def configure_logging() -> logging.Logger:
    """
    Configure project logger.

    Returns
    -------
    logging.Logger
    """

    logger = logging.getLogger(PROJECT_NAME)

    logger.setLevel(logging.INFO)

    logger.handlers.clear()

    formatter = logging.Formatter(

        "%(asctime)s | %(levelname)s | %(message)s"

    )

    file_handler = logging.FileHandler(LOG_FILE)

    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()

    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    logger.addHandler(console_handler)

    return logger


LOGGER = configure_logging()


# ==============================================================================
# Initialize Project
# ==============================================================================

create_directories()

set_random_seed()


LOGGER.info("=" * 80)

LOGGER.info(PROJECT_NAME)

LOGGER.info(MODULE_NAME)

LOGGER.info(f"Version : {MODULE_VERSION}")

LOGGER.info("=" * 80)


# ==============================================================================
# End of Part 3
# ==============================================================================

# ==============================================================================
# Validate Dataset Directories
# ==============================================================================

def validate_dataset() -> None:
    """
    Validate the required dataset directories.

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        If any dataset directory does not exist.
    """

    directories = {

        "Training Dataset": TRAIN_DIR,
        "Validation Dataset": VALIDATION_DIR,
        "Testing Dataset": TEST_DIR

    }

    for name, directory in directories.items():

        if not directory.exists():

            raise FileNotFoundError(

                f"{name} not found:\n{directory}"

            )


# ==============================================================================
# Display Configuration
# ==============================================================================

def display_configuration() -> None:
    """
    Display the complete project configuration.

    Returns
    -------
    None
    """

    LOGGER.info("=" * 80)

    LOGGER.info("PlantXAI-Net Training Configuration")

    LOGGER.info("=" * 80)

    LOGGER.info(f"Project Name        : {PROJECT_NAME}")

    LOGGER.info(f"Module             : {MODULE_NAME}")

    LOGGER.info(f"Framework          : {FRAMEWORK}")

    LOGGER.info(f"Model              : {MODEL_NAME}")

    LOGGER.info(f"Classes            : {NUM_CLASSES}")

    LOGGER.info(f"Image Size         : {IMAGE_SIZE}")

    LOGGER.info(f"Batch Size         : {BATCH_SIZE}")

    LOGGER.info(f"Epochs             : {NUM_EPOCHS}")

    LOGGER.info(f"Learning Rate      : {LEARNING_RATE}")

    LOGGER.info(f"Weight Decay       : {WEIGHT_DECAY}")

    LOGGER.info(f"Device             : {DEVICE}")

    LOGGER.info(f"Mixed Precision    : {USE_AMP}")

    LOGGER.info("=" * 80)


# ==============================================================================
# Initialize Configuration
# ==============================================================================

validate_dataset()

display_configuration()


print()

print("=" * 80)

print("PlantXAI-Net")

print("EfficientNet-B0 Training Pipeline")

print("Phase 1 Configuration Loaded Successfully")

print(f"Device : {DEVICE}")

print(f"Classes : {NUM_CLASSES}")

print("=" * 80)

print()


LOGGER.info("Phase 1 initialized successfully.")


# ==============================================================================
# End of Phase 1
# ==============================================================================

# ==============================================================================
#
# Phase 2
#
# Dataset Loading & Data Preparation
#
# Part 1
#
# Image Transformations
#
# ==============================================================================


# ==============================================================================
# ImageNet Normalization
# ==============================================================================

IMAGENET_MEAN = (

    0.485,
    0.456,
    0.406

)

IMAGENET_STD = (

    0.229,
    0.224,
    0.225

)


# ==============================================================================
# Training Image Transform
# ==============================================================================

TRAIN_TRANSFORM = transforms.Compose(

    [

        transforms.Resize(

            (

                IMAGE_SIZE,
                IMAGE_SIZE

            )

        ),

        transforms.RandomHorizontalFlip(

            p=0.5

        ),

        transforms.RandomRotation(

            degrees=15

        ),

        transforms.ColorJitter(

            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1

        ),

        transforms.ToTensor(),

        transforms.Normalize(

            mean=IMAGENET_MEAN,

            std=IMAGENET_STD

        )

    ]

)


# ==============================================================================
# Validation Image Transform
# ==============================================================================

VALIDATION_TRANSFORM = transforms.Compose(

    [

        transforms.Resize(

            (

                IMAGE_SIZE,
                IMAGE_SIZE

            )

        ),

        transforms.ToTensor(),

        transforms.Normalize(

            mean=IMAGENET_MEAN,

            std=IMAGENET_STD

        )

    ]

)


# ==============================================================================
# Test Image Transform
# ==============================================================================

TEST_TRANSFORM = transforms.Compose(

    [

        transforms.Resize(

            (

                IMAGE_SIZE,
                IMAGE_SIZE

            )

        ),

        transforms.ToTensor(),

        transforms.Normalize(

            mean=IMAGENET_MEAN,

            std=IMAGENET_STD

        )

    ]

)


# ==============================================================================
# End of Part 1
# ==============================================================================
# ==============================================================================
#
# Phase 2
#
# Dataset Loading & Data Preparation
#
# Part 2
#
# ImageFolder Dataset Initialization
#
# ==============================================================================


# ==============================================================================
# Create ImageFolder Datasets
# ==============================================================================

TRAIN_DATASET = datasets.ImageFolder(

    root=TRAIN_DIR,

    transform=TRAIN_TRANSFORM

)

VALIDATION_DATASET = datasets.ImageFolder(

    root=VALIDATION_DIR,

    transform=VALIDATION_TRANSFORM

)

TEST_DATASET = datasets.ImageFolder(

    root=TEST_DIR,

    transform=TEST_TRANSFORM

)


# ==============================================================================
# Dataset Information
# ==============================================================================

CLASS_NAMES = TRAIN_DATASET.classes

CLASS_TO_INDEX = TRAIN_DATASET.class_to_idx

INDEX_TO_CLASS = {

    index: class_name

    for class_name, index

    in CLASS_TO_INDEX.items()

}

NUM_CLASSES = len(CLASS_NAMES)


# ==============================================================================
# Dataset Statistics
# ==============================================================================

TRAIN_IMAGES = len(TRAIN_DATASET)

VALIDATION_IMAGES = len(VALIDATION_DATASET)

TEST_IMAGES = len(TEST_DATASET)

TOTAL_IMAGES = (

    TRAIN_IMAGES

    + VALIDATION_IMAGES

    + TEST_IMAGES

)


# ==============================================================================
# Verify Dataset Consistency
# ==============================================================================

if TRAIN_DATASET.classes != VALIDATION_DATASET.classes:

    raise RuntimeError(

        "Training and Validation datasets contain different class names."

    )


if TRAIN_DATASET.classes != TEST_DATASET.classes:

    raise RuntimeError(

        "Training and Test datasets contain different class names."

    )


# ==============================================================================
# Display Dataset Information
# ==============================================================================

LOGGER.info("=" * 80)

LOGGER.info("Dataset Successfully Loaded")

LOGGER.info("=" * 80)

LOGGER.info(f"Training Images     : {TRAIN_IMAGES}")

LOGGER.info(f"Validation Images   : {VALIDATION_IMAGES}")

LOGGER.info(f"Testing Images      : {TEST_IMAGES}")

LOGGER.info(f"Total Images        : {TOTAL_IMAGES}")

LOGGER.info(f"Number of Classes   : {NUM_CLASSES}")

LOGGER.info("=" * 80)


print()

print("=" * 80)

print("Dataset Successfully Loaded")

print(f"Training Images   : {TRAIN_IMAGES}")

print(f"Validation Images : {VALIDATION_IMAGES}")

print(f"Testing Images    : {TEST_IMAGES}")

print(f"Total Images      : {TOTAL_IMAGES}")

print(f"Classes           : {NUM_CLASSES}")

print("=" * 80)

print()


# ==============================================================================
# End of Part 2
# ==============================================================================
# ==============================================================================
#
# Phase 2
#
# Dataset Loading & Data Preparation
#
# Part 3
#
# Class Weight Calculation
#
# ==============================================================================


# ==============================================================================
# Calculate Class Distribution
# ==============================================================================

CLASS_COUNTS = np.bincount(

    TRAIN_DATASET.targets

)

LOGGER.info("=" * 80)

LOGGER.info("Class Distribution")

LOGGER.info("=" * 80)

for class_name, count in zip(

    CLASS_NAMES,

    CLASS_COUNTS

):

    LOGGER.info(

        f"{class_name:<45} {count:>6}"

    )

LOGGER.info("=" * 80)


# ==============================================================================
# Calculate Class Weights
# ==============================================================================

CLASS_WEIGHTS = (

    len(TRAIN_DATASET)

    /

    (

        NUM_CLASSES

        * CLASS_COUNTS

    )

)

CLASS_WEIGHTS = torch.tensor(

    CLASS_WEIGHTS,

    dtype=torch.float32,

    device=DEVICE

)


# ==============================================================================
# Display Class Weights
# ==============================================================================

LOGGER.info("Class Weights")

LOGGER.info("=" * 80)

for class_name, weight in zip(

    CLASS_NAMES,

    CLASS_WEIGHTS.cpu().numpy()

):

    LOGGER.info(

        f"{class_name:<45} {weight:.4f}"

    )

LOGGER.info("=" * 80)


print()

print("=" * 80)

print("Class Weights Successfully Calculated")

print("=" * 80)

print()


# ==============================================================================
# End of Part 3
# ==============================================================================

# ==============================================================================
#
# Phase 2
#
# Dataset Loading & Data Preparation
#
# Part 4
#
# DataLoader Configuration
#
# ==============================================================================


# ==============================================================================
# Training DataLoader
# ==============================================================================

TRAIN_LOADER = DataLoader(

    dataset=TRAIN_DATASET,

    batch_size=BATCH_SIZE,

    shuffle=SHUFFLE,

    num_workers=NUM_WORKERS,

    pin_memory=PIN_MEMORY,

    drop_last=False,

    persistent_workers=(NUM_WORKERS > 0)

)


# ==============================================================================
# Validation DataLoader
# ==============================================================================

VALIDATION_LOADER = DataLoader(

    dataset=VALIDATION_DATASET,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=NUM_WORKERS,

    pin_memory=PIN_MEMORY,

    drop_last=False,

    persistent_workers=(NUM_WORKERS > 0)

)


# ==============================================================================
# Test DataLoader
# ==============================================================================

TEST_LOADER = DataLoader(

    dataset=TEST_DATASET,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=NUM_WORKERS,

    pin_memory=PIN_MEMORY,

    drop_last=False,

    persistent_workers=(NUM_WORKERS > 0)

)

# ==============================================================================
# DataLoader Statistics
# ==============================================================================

TRAIN_BATCHES = len(TRAIN_LOADER)

VALIDATION_BATCHES = len(VALIDATION_LOADER)

TEST_BATCHES = len(TEST_LOADER)


LOGGER.info("=" * 80)

LOGGER.info("DataLoader Information")

LOGGER.info("=" * 80)

LOGGER.info(f"Training Batches     : {TRAIN_BATCHES}")

LOGGER.info(f"Validation Batches   : {VALIDATION_BATCHES}")

LOGGER.info(f"Testing Batches      : {TEST_BATCHES}")

LOGGER.info("=" * 80)


print()

print("=" * 80)

print("DataLoaders Created Successfully")

print("=" * 80)

print(f"Training Batches   : {TRAIN_BATCHES}")

print(f"Validation Batches : {VALIDATION_BATCHES}")

print(f"Testing Batches    : {TEST_BATCHES}")

print("=" * 80)

print()


# ==============================================================================
# End of Part 4
# ==============================================================================
# ==============================================================================
#
# ==============================================================================
#
# Phase 2
#
# Dataset Loading & Data Preparation
#
# Part 5
#
# Dataset Verification & Summary
#
# ==============================================================================


# ==============================================================================
# Verify Single DataLoader
# ==============================================================================

def verify_dataloader(
    loader: DataLoader,
    loader_name: str
) -> None:
    """
    Verify one dataloader.

    Parameters
    ----------
    loader : DataLoader

    loader_name : str

    Returns
    -------
    None
    """

    images, labels = next(iter(loader))

    LOGGER.info(
        f"{loader_name:<12} Images Shape : {tuple(images.shape)}"
    )

    LOGGER.info(
        f"{loader_name:<12} Labels Shape : {tuple(labels.shape)}"
    )


# ==============================================================================
# Verify All DataLoaders
# ==============================================================================

def verify_all_dataloaders() -> None:
    """
    Verify all dataloaders.

    Returns
    -------
    None
    """

    LOGGER.info("=" * 80)
    LOGGER.info("Verifying DataLoaders")
    LOGGER.info("=" * 80)

    verify_dataloader(
        TRAIN_LOADER,
        "Training"
    )

    verify_dataloader(
        VALIDATION_LOADER,
        "Validation"
    )

    verify_dataloader(
        TEST_LOADER,
        "Testing"
    )

    LOGGER.info("=" * 80)


# ==============================================================================
# Dataset Summary
# ==============================================================================

def display_dataset_summary() -> None:
    """
    Display dataset summary.

    Returns
    -------
    None
    """

    print()
    print("=" * 80)
    print("DATASET SUMMARY")
    print("=" * 80)

    print(f"Training Images      : {TRAIN_IMAGES}")
    print(f"Validation Images    : {VALIDATION_IMAGES}")
    print(f"Testing Images       : {TEST_IMAGES}")
    print(f"Total Images         : {TOTAL_IMAGES}")

    print()

    print(f"Training Batches     : {TRAIN_BATCHES}")
    print(f"Validation Batches   : {VALIDATION_BATCHES}")
    print(f"Testing Batches      : {TEST_BATCHES}")

    print()

    print(f"Classes              : {NUM_CLASSES}")
    print(f"Image Size           : {IMAGE_SIZE}")
    print(f"Batch Size           : {BATCH_SIZE}")

    print("=" * 80)
    print("Phase 2 Completed Successfully")
    print("=" * 80)
    print()

    LOGGER.info("Phase 2 completed successfully.")


# ==============================================================================
# End of Phase 2
# ==============================================================================

# ==============================================================================
#
# Phase 3
#
# EfficientNet-B0 Model Builder
#
# Part 1
#
# Load Pretrained Model
#
# ==============================================================================


# ==============================================================================
# Load EfficientNet-B0
# ==============================================================================

MODEL = models.efficientnet_b0(

    weights=models.EfficientNet_B0_Weights.DEFAULT

)


# ==============================================================================
# Display Model Information
# ==============================================================================

LOGGER.info("=" * 80)

LOGGER.info("Loading EfficientNet-B0")

LOGGER.info("=" * 80)

LOGGER.info(f"Model Name : {MODEL_NAME}")

LOGGER.info("Pretrained : ImageNet")

LOGGER.info("=" * 80)


print()

print("=" * 80)

print("EfficientNet-B0 Successfully Loaded")

print("Pretrained Weights : ImageNet")

print("=" * 80)

print()


# ==============================================================================
# End of Part 1
# ==============================================================================

# ==============================================================================
#
# Phase 3
#
# EfficientNet-B0 Model Builder
#
# Part 2
#
# Transfer Learning Configuration
#
# ==============================================================================


# ==============================================================================
# Freeze Feature Extractor
# ==============================================================================

for parameter in MODEL.parameters():

    parameter.requires_grad = False


# ==============================================================================
# Replace Classification Head
# ==============================================================================

IN_FEATURES = MODEL.classifier[1].in_features

MODEL.classifier = nn.Sequential(

    nn.Dropout(

        p=0.30

    ),

    nn.Linear(

        in_features=IN_FEATURES,

        out_features=NUM_CLASSES

    )

)


# ==============================================================================
# Unfreeze Classifier
# ==============================================================================

for parameter in MODEL.classifier.parameters():

    parameter.requires_grad = True


# ==============================================================================
# Move Model to Device
# ==============================================================================

MODEL = MODEL.to(DEVICE)


# ==============================================================================
# Trainable Parameters
# ==============================================================================

TOTAL_PARAMETERS = sum(

    parameter.numel()

    for parameter in MODEL.parameters()

)

TRAINABLE_PARAMETERS = sum(

    parameter.numel()

    for parameter in MODEL.parameters()

    if parameter.requires_grad

)

FROZEN_PARAMETERS = (

    TOTAL_PARAMETERS

    - TRAINABLE_PARAMETERS

)


# ==============================================================================
# Display Model Information
# ==============================================================================

LOGGER.info("=" * 80)

LOGGER.info("Transfer Learning Configuration")

LOGGER.info("=" * 80)

LOGGER.info(f"Input Features      : {IN_FEATURES}")

LOGGER.info(f"Output Classes      : {NUM_CLASSES}")

LOGGER.info(f"Total Parameters    : {TOTAL_PARAMETERS:,}")

LOGGER.info(f"Trainable           : {TRAINABLE_PARAMETERS:,}")

LOGGER.info(f"Frozen              : {FROZEN_PARAMETERS:,}")

LOGGER.info("=" * 80)


print()

print("=" * 80)

print("Transfer Learning Ready")

print(f"Output Classes : {NUM_CLASSES}")

print(f"Trainable Parameters : {TRAINABLE_PARAMETERS:,}")

print("=" * 80)

print()


# ==============================================================================
# End of Part 2
# ==============================================================================
# ==============================================================================
#
# Phase 3
#
# EfficientNet-B0 Model Builder
#
# Part 3
#
# Loss Function, Optimizer & Learning Rate Scheduler
#
# ==============================================================================


# ==============================================================================
# Loss Function
# ==============================================================================

CRITERION = nn.CrossEntropyLoss(

    weight=CLASS_WEIGHTS

)


# ==============================================================================
# Optimizer
# ==============================================================================

OPTIMIZER = optim.AdamW(

    params=filter(

        lambda parameter: parameter.requires_grad,

        MODEL.parameters()

    ),

    lr=LEARNING_RATE,

    weight_decay=WEIGHT_DECAY

)


# ==============================================================================
# Learning Rate Scheduler
# ==============================================================================

SCHEDULER = optim.lr_scheduler.ReduceLROnPlateau(

    optimizer=OPTIMIZER,

    mode="min",

    factor=0.1,

    patience=3,

    threshold=1e-4,

    min_lr=1e-6

)


# ==============================================================================
# Display Training Configuration
# ==============================================================================

LOGGER.info("=" * 80)

LOGGER.info("Training Configuration")

LOGGER.info("=" * 80)

LOGGER.info(f"Loss Function     : CrossEntropyLoss")

LOGGER.info(f"Optimizer         : AdamW")

LOGGER.info(f"Learning Rate     : {LEARNING_RATE}")

LOGGER.info(f"Weight Decay      : {WEIGHT_DECAY}")

LOGGER.info("Scheduler         : ReduceLROnPlateau")

LOGGER.info("=" * 80)


print()

print("=" * 80)

print("Training Configuration Ready")

print(f"Loss Function : CrossEntropyLoss")

print(f"Optimizer     : AdamW")

print(f"Learning Rate : {LEARNING_RATE}")

print("=" * 80)

print()


# ==============================================================================
# End of Part 3
# ==============================================================================
# ==============================================================================
#
# Phase 3
#
# EfficientNet-B0 Model Builder
#
# Part 4
#
# Mixed Precision, Early Stopping & Checkpoints
#
# ==============================================================================


# ==============================================================================
# Automatic Mixed Precision (AMP)
# ==============================================================================

SCALER = torch.cuda.amp.GradScaler(

    enabled=USE_AMP

)


# ==============================================================================
# Early Stopping Configuration
# ==============================================================================

EARLY_STOPPING_COUNTER = 0

BEST_VALIDATION_LOSS = float("inf")

BEST_VALIDATION_ACCURACY = 0.0

CURRENT_EPOCH = 0


# ==============================================================================
# Checkpoint Files
# ==============================================================================

BEST_MODEL_PATH = (

    CHECKPOINT_DIR

    / "best_model.pth"

)

LAST_CHECKPOINT_PATH = (

    CHECKPOINT_DIR

    / "last_checkpoint.pth"

)

TRAINING_HISTORY_PATH = (

    METRIC_DIR

    / "training_history.csv"

)


# ==============================================================================
# Training History
# ==============================================================================

TRAIN_LOSS_HISTORY = []

VALIDATION_LOSS_HISTORY = []

TRAIN_ACCURACY_HISTORY = []

VALIDATION_ACCURACY_HISTORY = []


# ==============================================================================
# Display Configuration
# ==============================================================================

LOGGER.info("=" * 80)

LOGGER.info("Training Configuration")

LOGGER.info("=" * 80)

LOGGER.info(f"Mixed Precision        : {USE_AMP}")

LOGGER.info(f"Early Stopping         : {EARLY_STOPPING_PATIENCE}")

LOGGER.info(f"Checkpoint Directory   : {CHECKPOINT_DIR}")

LOGGER.info(f"Best Model Path        : {BEST_MODEL_PATH}")

LOGGER.info(f"Last Checkpoint Path   : {LAST_CHECKPOINT_PATH}")

LOGGER.info("=" * 80)


print()

print("=" * 80)

print("Training Infrastructure Ready")

print()

print(f"AMP Enabled        : {USE_AMP}")

print(f"Early Stopping     : {EARLY_STOPPING_PATIENCE}")

print()

print("Checkpoint System Initialized")

print("=" * 80)

print()


# ==============================================================================
# End of Part 4
# ==============================================================================
# ==============================================================================
#
# Phase 3
#
# EfficientNet-B0 Model Builder
#
# Part 5
#
# Model Summary & Verification
#
# ==============================================================================


# ==============================================================================
# Display Model Architecture Summary
# ==============================================================================

LOGGER.info("=" * 80)

LOGGER.info("Model Summary")

LOGGER.info("=" * 80)

LOGGER.info(f"Model Name              : {MODEL_NAME}")

LOGGER.info(f"Framework               : {FRAMEWORK}")

LOGGER.info(f"Total Classes           : {NUM_CLASSES}")

LOGGER.info(f"Input Image Size        : {IMAGE_SIZE} x {IMAGE_SIZE}")

LOGGER.info(f"Batch Size              : {BATCH_SIZE}")

LOGGER.info(f"Learning Rate           : {LEARNING_RATE}")

LOGGER.info(f"Optimizer               : AdamW")

LOGGER.info(f"Loss Function           : CrossEntropyLoss")

LOGGER.info(f"Device                  : {DEVICE}")

LOGGER.info("=" * 80)


# ==============================================================================
# Verify Forward Pass
# ==============================================================================

MODEL.eval()

with torch.no_grad():

    dummy_input = torch.randn(

        1,

        3,

        IMAGE_SIZE,

        IMAGE_SIZE,

        device=DEVICE

    )

    dummy_output = MODEL(dummy_input)


LOGGER.info(

    f"Dummy Output Shape : {tuple(dummy_output.shape)}"

)


if dummy_output.shape[1] != NUM_CLASSES:

    raise RuntimeError(

        "Model output dimension does not match the number of classes."

    )


LOGGER.info(

    "Forward pass verification successful."

)


# ==============================================================================
# Display Summary
# ==============================================================================

print()

print("=" * 80)

print("MODEL CONFIGURATION")

print("=" * 80)

print(f"Model              : {MODEL_NAME}")

print(f"Framework          : {FRAMEWORK}")

print(f"Classes            : {NUM_CLASSES}")

print(f"Device             : {DEVICE}")

print(f"Trainable Params   : {TRAINABLE_PARAMETERS:,}")

print(f"Frozen Params      : {FROZEN_PARAMETERS:,}")

print()

print("Forward Pass       : SUCCESS")

print()

print("Phase 3 Completed Successfully")

print("=" * 80)

print()


LOGGER.info("=" * 80)

LOGGER.info("Phase 3 Completed Successfully")

LOGGER.info("=" * 80)


# ==============================================================================
# End of Phase 3
# ==============================================================================
# ==============================================================================
# ==============================================================================
#
# Phase 4
#
# Training Engine
#
# Part 1
#
# Train One Epoch
#
# ==============================================================================


def train_one_epoch(
    epoch: int
) -> tuple[float, float]:

    """
    Train model for one epoch.

    Parameters
    ----------
    epoch : int

    Returns
    -------
    tuple
        (train_loss, train_accuracy)
    """

    MODEL.train()

    running_loss = 0.0

    correct_predictions = 0

    total_samples = 0

    progress_bar = tqdm(

        TRAIN_LOADER,

        desc=f"Training Epoch {epoch + 1}/{NUM_EPOCHS}",

        leave=False

    )

    for images, labels in progress_bar:

        images = images.to(

            DEVICE,

            non_blocking=True

        )

        labels = labels.to(

            DEVICE,

            non_blocking=True

        )

        OPTIMIZER.zero_grad(

            set_to_none=True

        )

        with torch.cuda.amp.autocast(

            enabled=USE_AMP

        ):

            outputs = MODEL(

                images

            )

            loss = CRITERION(

                outputs,

                labels

            )

        SCALER.scale(

            loss

        ).backward()

        SCALER.step(

            OPTIMIZER

        )

        SCALER.update()

        predictions = torch.argmax(

            outputs,

            dim=1

        )

        correct_predictions += (

            predictions == labels

        ).sum().item()

        total_samples += labels.size(0)

        running_loss += (

            loss.item()

            * labels.size(0)

        )

        progress_bar.set_postfix(

            {

                "Loss": f"{loss.item():.4f}"

            }

        )

    epoch_loss = (

        running_loss

        / total_samples

    )

    epoch_accuracy = (

        correct_predictions

        / total_samples

    )

    return (

        epoch_loss,

        epoch_accuracy

    )


# ==============================================================================
# End of Part 1
# ==============================================================================
# ==============================================================================
#
# Phase 4
#
# Training Engine
#
# Part 2
#
# Validate One Epoch
#
# ==============================================================================

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def validate_one_epoch(
    epoch: int
) -> tuple[float, float, float, float, float]:

    """
    Validate the model for one epoch.

    Parameters
    ----------
    epoch : int

    Returns
    -------
    tuple
        (
            validation_loss,
            validation_accuracy,
            precision,
            recall,
            f1_score
        )
    """

    MODEL.eval()

    running_loss = 0.0

    all_predictions = []

    all_labels = []

    progress_bar = tqdm(

        VALIDATION_LOADER,

        desc=f"Validation Epoch {epoch + 1}/{NUM_EPOCHS}",

        leave=False

    )

    with torch.no_grad():

        for images, labels in progress_bar:

            images = images.to(

                DEVICE,

                non_blocking=True

            )

            labels = labels.to(

                DEVICE,

                non_blocking=True

            )

            with torch.cuda.amp.autocast(

                enabled=USE_AMP

            ):

                outputs = MODEL(

                    images

                )

                loss = CRITERION(

                    outputs,

                    labels

                )

            predictions = torch.argmax(

                outputs,

                dim=1

            )

            running_loss += (

                loss.item()

                * labels.size(0)

            )

            all_predictions.extend(

                predictions.cpu().tolist()

            )

            all_labels.extend(

                labels.cpu().tolist()

            )

            progress_bar.set_postfix(

                {

                    "Loss": f"{loss.item():.4f}"

                }

            )

    validation_loss = (

        running_loss

        / len(VALIDATION_DATASET)

    )

    validation_accuracy = accuracy_score(

        all_labels,

        all_predictions

    )

    precision = precision_score(

        all_labels,

        all_predictions,

        average="weighted",

        zero_division=0

    )

    recall = recall_score(

        all_labels,

        all_predictions,

        average="weighted",

        zero_division=0

    )

    f1 = f1_score(

        all_labels,

        all_predictions,

        average="weighted",

        zero_division=0

    )

    return (

        validation_loss,

        validation_accuracy,

        precision,

        recall,

        f1

    )


# ==============================================================================
# End of Part 2
# ==============================================================================
# ==============================================================================
# ==============================================================================
#
# Phase 4
#
# Training Engine
#
# Part 3
#
# Checkpoint Management
#
# ==============================================================================


# ==============================================================================
# Save Checkpoint
# ==============================================================================

def save_checkpoint(
    epoch: int,
    validation_loss: float,
    validation_accuracy: float
) -> None:
    """
    Save model checkpoint.

    Parameters
    ----------
    epoch : int
        Current epoch number.

    validation_loss : float
        Validation loss.

    validation_accuracy : float
        Validation accuracy.

    Returns
    -------
    None
    """

    global BEST_VALIDATION_LOSS
    global BEST_VALIDATION_ACCURACY

    checkpoint = {

        # ----------------------------------------------------------
        # Training Progress
        # ----------------------------------------------------------

        "epoch": epoch,

        # ----------------------------------------------------------
        # Model States
        # ----------------------------------------------------------

        "model_state_dict": MODEL.state_dict(),

        "optimizer_state_dict": OPTIMIZER.state_dict(),

        "scheduler_state_dict": SCHEDULER.state_dict(),

        # ----------------------------------------------------------
        # Best Metrics
        # ----------------------------------------------------------

        "validation_loss": validation_loss,

        "validation_accuracy": validation_accuracy,

        # ----------------------------------------------------------
        # Training History
        # ----------------------------------------------------------

        "train_loss_history": TRAIN_LOSS_HISTORY,

        "validation_loss_history": VALIDATION_LOSS_HISTORY,

        "train_accuracy_history": TRAIN_ACCURACY_HISTORY,

        "validation_accuracy_history": VALIDATION_ACCURACY_HISTORY

    }

    # ----------------------------------------------------------
    # Save Latest Checkpoint
    # ----------------------------------------------------------

    torch.save(

        checkpoint,

        LAST_CHECKPOINT_PATH

    )

    # ----------------------------------------------------------
    # Save Best Model
    # ----------------------------------------------------------

    is_best_model = (

        validation_accuracy > BEST_VALIDATION_ACCURACY

    ) or (

        validation_accuracy == BEST_VALIDATION_ACCURACY

        and

        validation_loss < BEST_VALIDATION_LOSS

    )

    if is_best_model:

        BEST_VALIDATION_ACCURACY = validation_accuracy

        BEST_VALIDATION_LOSS = validation_loss

        torch.save(

            checkpoint,

            BEST_MODEL_PATH

        )

        LOGGER.info(

            "Best model saved."

        )


# ==============================================================================
# Load Checkpoint
# ==============================================================================

def load_checkpoint() -> int:
    """
    Load latest checkpoint.

    Returns
    -------
    int
        Starting epoch.
    """

    global BEST_VALIDATION_LOSS
    global BEST_VALIDATION_ACCURACY

    global TRAIN_LOSS_HISTORY
    global VALIDATION_LOSS_HISTORY

    global TRAIN_ACCURACY_HISTORY
    global VALIDATION_ACCURACY_HISTORY

    # ----------------------------------------------------------
    # Check Availability
    # ----------------------------------------------------------

    if not LAST_CHECKPOINT_PATH.exists():

        LOGGER.info(

            "No previous checkpoint found."

        )

        return 0

    # ----------------------------------------------------------
    # Load Checkpoint
    # ----------------------------------------------------------

    checkpoint = torch.load(

        LAST_CHECKPOINT_PATH,

        map_location=DEVICE

    )

    # ----------------------------------------------------------
    # Restore Model
    # ----------------------------------------------------------

    MODEL.load_state_dict(

        checkpoint["model_state_dict"]

    )

    OPTIMIZER.load_state_dict(

        checkpoint["optimizer_state_dict"]

    )

    SCHEDULER.load_state_dict(

        checkpoint["scheduler_state_dict"]

    )

    # ----------------------------------------------------------
    # Restore Best Metrics
    # ----------------------------------------------------------

    BEST_VALIDATION_LOSS = checkpoint["validation_loss"]

    BEST_VALIDATION_ACCURACY = checkpoint["validation_accuracy"]

    # ----------------------------------------------------------
    # Restore Training History
    # ----------------------------------------------------------

    TRAIN_LOSS_HISTORY = checkpoint.get(

        "train_loss_history",

        []

    )

    VALIDATION_LOSS_HISTORY = checkpoint.get(

        "validation_loss_history",

        []

    )

    TRAIN_ACCURACY_HISTORY = checkpoint.get(

        "train_accuracy_history",

        []

    )

    VALIDATION_ACCURACY_HISTORY = checkpoint.get(

        "validation_accuracy_history",

        []

    )

    # ----------------------------------------------------------
    # Logger
    # ----------------------------------------------------------

    LOGGER.info(

        f"Checkpoint Loaded Successfully (Epoch {checkpoint['epoch'] + 1})"

    )

    LOGGER.info(

        f"Recovered {len(TRAIN_LOSS_HISTORY)} Epoch(s) of Training History."

    )

    return checkpoint["epoch"] + 1


# ==============================================================================
# End of Part 3
# ==============================================================================
#
# Phase 4
#
# Training Engine
#
# Part 4
#
# Train Model
#
# ==============================================================================


def train_model() -> None:
    """
    Train the model.

    Returns
    -------
    None
    """

    global EARLY_STOPPING_COUNTER

    start_epoch = load_checkpoint()

    LOGGER.info("=" * 80)
    LOGGER.info("Training Started")
    LOGGER.info("=" * 80)

    for epoch in range(

        start_epoch,

        NUM_EPOCHS

    ):

        epoch_start_time = time.time()

        # ----------------------------------------------------------
        # Training
        # ----------------------------------------------------------

        train_loss, train_accuracy = train_one_epoch(

            epoch

        )

        # ----------------------------------------------------------
        # Validation
        # ----------------------------------------------------------

        (
            validation_loss,
            validation_accuracy,
            precision,
            recall,
            f1

        ) = validate_one_epoch(

            epoch

        )

        # ----------------------------------------------------------
        # Scheduler
        # ----------------------------------------------------------

        SCHEDULER.step(

            validation_loss

        )

        # ----------------------------------------------------------
        # Store History
        # ----------------------------------------------------------

        TRAIN_LOSS_HISTORY.append(

            train_loss

        )

        VALIDATION_LOSS_HISTORY.append(

            validation_loss

        )

        TRAIN_ACCURACY_HISTORY.append(

            train_accuracy

        )

        VALIDATION_ACCURACY_HISTORY.append(

            validation_accuracy

        )

        # ----------------------------------------------------------
        # Check Improvement
        # ----------------------------------------------------------

        is_improved = (

            validation_accuracy > BEST_VALIDATION_ACCURACY

        ) or (

            validation_accuracy == BEST_VALIDATION_ACCURACY

            and

            validation_loss < BEST_VALIDATION_LOSS

        )

        if is_improved:

            EARLY_STOPPING_COUNTER = 0

        else:

            EARLY_STOPPING_COUNTER += 1

        # ----------------------------------------------------------
        # Save Checkpoint
        # ----------------------------------------------------------

        save_checkpoint(

            epoch,

            validation_loss,

            validation_accuracy

        )

        epoch_time = (

            time.time()

            - epoch_start_time

        )

        # ----------------------------------------------------------
        # Logger
        # ----------------------------------------------------------

        LOGGER.info(

            "-" * 80

        )

        LOGGER.info(

            f"Epoch {epoch + 1}/{NUM_EPOCHS}"

        )

        LOGGER.info(

            f"Train Loss          : {train_loss:.4f}"

        )

        LOGGER.info(

            f"Validation Loss     : {validation_loss:.4f}"

        )

        LOGGER.info(

            f"Train Accuracy      : {train_accuracy:.4f}"

        )

        LOGGER.info(

            f"Validation Accuracy : {validation_accuracy:.4f}"

        )

        LOGGER.info(

            f"Precision           : {precision:.4f}"

        )

        LOGGER.info(

            f"Recall              : {recall:.4f}"

        )

        LOGGER.info(

            f"F1 Score            : {f1:.4f}"

        )

        LOGGER.info(

            f"Epoch Time          : {epoch_time:.2f} sec"

        )

        LOGGER.info(

            "-" * 80

        )

        # ----------------------------------------------------------
        # Early Stopping
        # ----------------------------------------------------------

        if (

            EARLY_STOPPING_COUNTER

            >=

            EARLY_STOPPING_PATIENCE

        ):

            LOGGER.info(

                "Early Stopping Triggered."

            )

            break

    save_training_results()

    verify_training()

    LOGGER.info("=" * 80)

    LOGGER.info("Training Completed")

    LOGGER.info("=" * 80)


# ==============================================================================
# End of Part 4
# ==============================================================================
# ==============================================================================
#
# Phase 4
#
# Training Engine
#
# Part 5
#
# Save Training Results
#
# ==============================================================================


def save_training_results() -> None:
    """
    Save training history and performance plots.

    Returns
    -------
    None
    """

    history = pd.DataFrame(

        {

            "Epoch": list(

                range(

                    1,

                    len(TRAIN_LOSS_HISTORY) + 1

                )

            ),

            "Train Loss": TRAIN_LOSS_HISTORY,

            "Validation Loss": VALIDATION_LOSS_HISTORY,

            "Train Accuracy": TRAIN_ACCURACY_HISTORY,

            "Validation Accuracy": VALIDATION_ACCURACY_HISTORY

        }

    )

    history.to_csv(

        TRAINING_HISTORY_PATH,

        index=False

    )

    LOGGER.info(

        "Training history saved."

    )

    # ----------------------------------------------------------
    # Loss Curve
    # ----------------------------------------------------------

    plt.figure(

        figsize=(8, 5)

    )

    plt.plot(

        history["Epoch"],

        history["Train Loss"],

        label="Train"

    )

    plt.plot(

        history["Epoch"],

        history["Validation Loss"],

        label="Validation"

    )

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.title("Loss Curve")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(

        PLOT_DIR / "loss_curve.png"

    )

    plt.close()

    # ----------------------------------------------------------
    # Accuracy Curve
    # ----------------------------------------------------------

    plt.figure(

        figsize=(8, 5)

    )

    plt.plot(

        history["Epoch"],

        history["Train Accuracy"],

        label="Train"

    )

    plt.plot(

        history["Epoch"],

        history["Validation Accuracy"],

        label="Validation"

    )

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy")

    plt.title("Accuracy Curve")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(

        PLOT_DIR / "accuracy_curve.png"

    )

    plt.close()

    LOGGER.info(

        "Training plots generated successfully."


    )


# ==============================================================================
# End of Part 5
# ==============================================================================
# ==============================================================================
#
# Phase 4
#
# Training Engine
#
# Part 6
#
# Training Verification
#
# ==============================================================================


def verify_training() -> None:
    """
    Display final training summary.

    Returns
    -------
    None
    """

    LOGGER.info("=" * 80)
    LOGGER.info("Training Summary")
    LOGGER.info("=" * 80)

    LOGGER.info(
        f"Epochs Completed          : {len(TRAIN_LOSS_HISTORY)}"
    )

    LOGGER.info(
        f"Best Validation Accuracy  : {BEST_VALIDATION_ACCURACY:.4f}"
    )

    LOGGER.info(
        f"Best Validation Loss      : {BEST_VALIDATION_LOSS:.4f}"
    )

    LOGGER.info(
        f"Best Model               : {BEST_MODEL_PATH}"
    )

    LOGGER.info(
        f"Training History         : {TRAINING_HISTORY_PATH}"
    )

    LOGGER.info("=" * 80)

    print()

    print("=" * 80)

    print("TRAINING COMPLETED SUCCESSFULLY")

    print("=" * 80)

    print(
        f"Epochs Completed      : {len(TRAIN_LOSS_HISTORY)}"
    )

    print(
        f"Best Validation Accuracy : {BEST_VALIDATION_ACCURACY:.4f}"
    )

    print(
        f"Best Validation Loss     : {BEST_VALIDATION_LOSS:.4f}"
    )

    print()

    print(
        f"Best Model Saved At : {BEST_MODEL_PATH}"
    )

    print(
        f"History Saved At    : {TRAINING_HISTORY_PATH}"
    )

    print("=" * 80)

    print("Phase 4 Completed Successfully")

    print("=" * 80)

    print()


# ==============================================================================
# End of Phase 4
# ==============================================================================
# ==============================================================================
## ==============================================================================
# ==============================================================================
#
# PlantXAI-Net
#
# Phase 5
#
# Evaluation Module
#
# Part 1
#
# Model Evaluation + Existing Checkpoint Verification
#
# ==============================================================================


# ==============================================================================
# Evaluate Model
# ==============================================================================

def evaluate_model() -> tuple:

    """
    Evaluate the current loaded model on the complete test dataset.

    Returns
    -------
    tuple

        (
            test_loss,
            test_accuracy,
            predictions,
            labels
        )
    """

    LOGGER.info("=" * 80)

    LOGGER.info("Evaluating Model")

    LOGGER.info("=" * 80)


    # --------------------------------------------------------------------------
    # Evaluation Mode
    # --------------------------------------------------------------------------

    MODEL.eval()


    # --------------------------------------------------------------------------
    # Initialize Variables
    # --------------------------------------------------------------------------

    running_loss = 0.0

    all_predictions = []

    all_labels = []


    # --------------------------------------------------------------------------
    # Test Progress Bar
    # --------------------------------------------------------------------------

    progress_bar = tqdm(

        TEST_LOADER,

        desc="Testing",

        leave=False

    )


    # ==========================================================================
    # Test Loop
    # ==========================================================================

    with torch.no_grad():

        for images, labels in progress_bar:


            # ------------------------------------------------------------------
            # Move Images and Labels to Device
            # ------------------------------------------------------------------

            images = images.to(

                DEVICE,

                non_blocking=True

            )


            labels = labels.to(

                DEVICE,

                non_blocking=True

            )


            # ------------------------------------------------------------------
            # Forward Pass
            # ------------------------------------------------------------------

            with torch.cuda.amp.autocast(

                enabled=USE_AMP

            ):

                outputs = MODEL(

                    images

                )


                loss = CRITERION(

                    outputs,

                    labels

                )


            # ------------------------------------------------------------------
            # Predicted Class
            # ------------------------------------------------------------------

            predictions = torch.argmax(

                outputs,

                dim=1

            )


            # ------------------------------------------------------------------
            # Accumulate Test Loss
            # ------------------------------------------------------------------

            running_loss += (

                loss.item()

                *

                labels.size(0)

            )


            # ------------------------------------------------------------------
            # Store Predictions
            # ------------------------------------------------------------------

            all_predictions.extend(

                predictions.cpu().tolist()

            )


            # ------------------------------------------------------------------
            # Store Ground-Truth Labels
            # ------------------------------------------------------------------

            all_labels.extend(

                labels.cpu().tolist()

            )


            # ------------------------------------------------------------------
            # Update Progress Bar
            # ------------------------------------------------------------------

            progress_bar.set_postfix(

                {

                    "Loss": f"{loss.item():.4f}"

                }

            )


    # ==========================================================================
    # Calculate Test Loss
    # ==========================================================================

    test_loss = (

        running_loss

        /

        len(TEST_DATASET)

    )


    # ==========================================================================
    # Calculate Test Accuracy
    # ==========================================================================

    test_accuracy = accuracy_score(

        all_labels,

        all_predictions

    )


    # ==========================================================================
    # Log Results
    # ==========================================================================

    LOGGER.info(

        f"Test Loss : {test_loss:.4f}"

    )


    LOGGER.info(

        f"Test Accuracy : {test_accuracy:.4f}"

    )


    # ==========================================================================
    # Return Evaluation Results
    # ==========================================================================

    return (

        test_loss,

        test_accuracy,

        all_predictions,

        all_labels

    )


# ==============================================================================
# Verify Existing Checkpoint
# ==============================================================================

def verify_existing_checkpoint() -> None:

    """
    Load an already-trained checkpoint and evaluate it on the complete
    test dataset.

    IMPORTANT
    ---------
    This function performs NO training.

    By default, the existing last_checkpoint.pth is evaluated so that
    the previously completed 8-epoch model can be verified.
    """


    # ==========================================================================
    # Check Last Checkpoint
    # ==========================================================================

    if not LAST_CHECKPOINT_PATH.exists():

        raise FileNotFoundError(

            f"Last checkpoint not found: "
            f"{LAST_CHECKPOINT_PATH}"

        )


    # ==========================================================================
    # Load Existing Checkpoint
    # ==========================================================================

    checkpoint = torch.load(

        LAST_CHECKPOINT_PATH,

        map_location=DEVICE

    )


    # ==========================================================================
    # Validate Checkpoint Structure
    # ==========================================================================

    if "model_state_dict" not in checkpoint:

        raise KeyError(

            "model_state_dict not found in "
            "last_checkpoint.pth."

        )


    # ==========================================================================
    # Load Model Weights
    # ==========================================================================

    MODEL.load_state_dict(

        checkpoint["model_state_dict"]

    )


    MODEL.to(DEVICE)

    MODEL.eval()


    # ==========================================================================
    # Display Checkpoint Information
    # ==========================================================================

    print()

    print("=" * 80)

    print(
        "EXISTING CHECKPOINT VERIFICATION"
    )

    print("=" * 80)

    print()


    print(

        f"Checkpoint       : "
        f"{LAST_CHECKPOINT_PATH}"

    )


    print(

        f"Checkpoint Epoch : "
        f"{checkpoint.get('epoch', 'N/A')}"

    )


    if "epoch" in checkpoint:

        actual_training_epoch = (
            int(checkpoint["epoch"])
            +
            1
        )

        print(

            f"Training Epoch   : "
            f"{actual_training_epoch}"

        )


    if "validation_accuracy" in checkpoint:

        print(

            f"Validation Accuracy : "
            f"{checkpoint['validation_accuracy'] * 100:.4f}%"

        )


    if "validation_loss" in checkpoint:

        print(

            f"Validation Loss     : "
            f"{checkpoint['validation_loss']:.6f}"

        )


    print()


    # ==========================================================================
    # Evaluate Loaded Checkpoint
    # ==========================================================================

    (
        test_loss,
        test_accuracy,
        predictions,
        labels

    ) = evaluate_model()


    # ==========================================================================
    # Calculate Prediction Counts
    # ==========================================================================

    total_images = len(

        labels

    )


    correct_predictions = sum(

        int(

            prediction == label

        )

        for prediction, label

        in zip(

            predictions,

            labels

        )

    )


    incorrect_predictions = (

        total_images

        -

        correct_predictions

    )


    # ==========================================================================
    # Final Verification Output
    # ==========================================================================

    print()

    print("=" * 80)

    print(
        "CHECKPOINT VERIFICATION RESULT"
    )

    print("=" * 80)

    print()


    print(

        f"Images Evaluated       : "
        f"{total_images}"

    )


    print(

        f"Correct Predictions    : "
        f"{correct_predictions}"

    )


    print(

        f"Incorrect Predictions  : "
        f"{incorrect_predictions}"

    )


    print(

        f"Test Accuracy          : "
        f"{test_accuracy * 100:.4f}%"

    )


    print(

        f"Test Loss              : "
        f"{test_loss:.4f}"

    )


    print()


    print("=" * 80)

    print(
        "Checkpoint Verification Completed Successfully"
    )

    print("=" * 80)

    print()


# ==============================================================================
# End of Phase 5 - Part 1
# ==============================================================================
#
# Phase 5
#
# Evaluation Module
#
# Part 2
#
# Classification Report
#
# ==============================================================================


def generate_classification_report(
    predictions: list,
    labels: list
) -> None:
    """
    Generate and save classification report.

    Parameters
    ----------
    predictions : list

    labels : list

    Returns
    -------
    None
    """

    LOGGER.info("=" * 80)
    LOGGER.info("Generating Classification Report")
    LOGGER.info("=" * 80)

    report = classification_report(

        labels,

        predictions,

        target_names=CLASS_NAMES,

        digits=4,

        zero_division=0

    )

    report_path = REPORT_DIR / "classification_report.txt"

    with open(

        report_path,

        "w",

        encoding="utf-8"

    ) as file:

        file.write(report)

    LOGGER.info(
        f"Classification report saved to : {report_path}"
    )

    LOGGER.info("=" * 80)
    LOGGER.info("Classification Report Generated Successfully")
    LOGGER.info("=" * 80)

    print()
    print("=" * 80)
    print("CLASSIFICATION REPORT GENERATED")
    print("=" * 80)
    print(f"Saved To : {report_path}")
    print("=" * 80)
    print()


# ==============================================================================
# End of Part 2
# ==============================================================================
#
# Phase 5
#
# Evaluation Module
#
# Part 3
#
# Confusion Matrix
#
# ==============================================================================

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


def generate_confusion_matrix(
    predictions: list,
    labels: list
) -> None:
    """
    Generate and save confusion matrix.

    Parameters
    ----------
    predictions : list

    labels : list

    Returns
    -------
    None
    """

    LOGGER.info("=" * 80)

    LOGGER.info("Generating Confusion Matrix")

    LOGGER.info("=" * 80)

    matrix = confusion_matrix(

        labels,

        predictions

    )

    figure, axis = plt.subplots(

        figsize=(14, 14)

    )

    display = ConfusionMatrixDisplay(

        confusion_matrix=matrix,

        display_labels=CLASS_NAMES

    )

    display.plot(

        ax=axis,

        xticks_rotation=90,

        colorbar=True

    )

    plt.title(

        "Confusion Matrix"

    )

    plt.tight_layout()

    output_path = (

        PLOT_DIR

        /

        "confusion_matrix.png"

    )

    plt.savefig(

        output_path,

        dpi=300,

        bbox_inches="tight"

    )

    plt.close()

    LOGGER.info(

        f"Confusion Matrix saved to : {output_path}"

    )


# ==============================================================================
# End of Part 3
# ==============================================================================
# ==============================================================================
#
# Phase 5
#
# Evaluation Module
#
# Part 4
#
# Save Prediction Results
#
# ==============================================================================


def save_prediction_results(
    predictions: list,
    labels: list
) -> None:
    """
    Save prediction results to CSV.

    Parameters
    ----------
    predictions : list

    labels : list

    Returns
    -------
    None
    """

    LOGGER.info("=" * 80)

    LOGGER.info("Saving Prediction Results")

    LOGGER.info("=" * 80)

    prediction_dataframe = pd.DataFrame(

        {

            "Actual Label": [

                CLASS_NAMES[label]

                for label in labels

            ],

            "Predicted Label": [

                CLASS_NAMES[prediction]

                for prediction in predictions

            ],

            "Correct Prediction": [

                actual == prediction

                for actual, prediction

                in zip(

                    labels,

                    predictions

                )

            ]

        }

    )

    output_path = (

        REPORT_DIR

        /

        "prediction_results.csv"

    )

    prediction_dataframe.to_csv(

        output_path,

        index=False

    )

    LOGGER.info(

        f"Prediction results saved to : {output_path}"

    )


# ==============================================================================
# End of Part 4
# ==============================================================================
# ==============================================================================
#
# Phase 5
#
# Evaluation Module
#
# Part 5
#
# Evaluation Visualizations
#
# ==============================================================================


def generate_evaluation_visualizations(
    predictions: list,
    labels: list
) -> None:
    """
    Generate evaluation visualizations.

    Parameters
    ----------
    predictions : list

    labels : list

    Returns
    -------
    None
    """

    LOGGER.info("=" * 80)

    LOGGER.info("Generating Evaluation Visualizations")

    LOGGER.info("=" * 80)

    # ----------------------------------------------------------
    # Per-Class Accuracy
    # ----------------------------------------------------------

    class_accuracy = []

    for class_index in range(len(CLASS_NAMES)):

        total = 0

        correct = 0

        for actual, predicted in zip(labels, predictions):

            if actual == class_index:

                total += 1

                if predicted == actual:

                    correct += 1

        if total == 0:

            class_accuracy.append(0)

        else:

            class_accuracy.append(

                100 * correct / total

            )

    plt.figure(

        figsize=(16, 8)

    )

    plt.bar(

        CLASS_NAMES,

        class_accuracy

    )

    plt.xticks(

        rotation=90

    )

    plt.ylabel(

        "Accuracy (%)"

    )

    plt.xlabel(

        "Classes"

    )

    plt.title(

        "Per-Class Accuracy"

    )

    plt.grid(

        axis="y"

    )

    plt.tight_layout()

    plt.savefig(

        PLOT_DIR /

        "per_class_accuracy.png",

        dpi=300

    )

    plt.close()

    LOGGER.info(

        "Per-Class Accuracy plot saved."

    )


# ==============================================================================
# End of Part 5
# ==============================================================================
# ==============================================================================
# ==============================================================================
# ==============================================================================
#
# Phase 5
#
# Evaluation Module
#
# Part 6
#
# Complete Evaluation Pipeline
#
# ==============================================================================


def verify_evaluation(
    test_loss: float,
    test_accuracy: float
) -> None:
    """
    Verify evaluation results.

    Parameters
    ----------
    test_loss : float

    test_accuracy : float

    Returns
    -------
    None
    """

    LOGGER.info("=" * 80)
    LOGGER.info("Evaluation Summary")
    LOGGER.info("=" * 80)

    LOGGER.info(f"Test Loss          : {test_loss:.4f}")
    LOGGER.info(f"Test Accuracy      : {test_accuracy:.4f}")
    LOGGER.info(f"Classification     : {REPORT_DIR / 'classification_report.txt'}")
    LOGGER.info(f"Predictions        : {REPORT_DIR / 'prediction_results.csv'}")
    LOGGER.info(f"Confusion Matrix   : {PLOT_DIR / 'confusion_matrix.png'}")
    LOGGER.info(f"Metrics            : {METRIC_DIR / 'metrics.csv'}")

    LOGGER.info("=" * 80)


# ==============================================================================
# Save Final Metrics
# ==============================================================================

def save_final_metrics(
    test_loss: float,
    test_accuracy: float
) -> None:
    """
    Save final project metrics.
    """

    metrics = pd.DataFrame(

        {

            "Metric": [

                "Model",

                "Classes",

                "Epochs",

                "Batch Size",

                "Learning Rate",

                "Best Validation Accuracy",

                "Best Validation Loss",

                "Final Test Accuracy",

                "Final Test Loss"

            ],

            "Value": [

                MODEL_NAME,

                NUM_CLASSES,

                NUM_EPOCHS,

                BATCH_SIZE,

                LEARNING_RATE,

                BEST_VALIDATION_ACCURACY,

                BEST_VALIDATION_LOSS,

                test_accuracy,

                test_loss

            ]

        }

    )

    output_path = METRIC_DIR / "metrics.csv"

    metrics.to_csv(

        output_path,

        index=False

    )

    LOGGER.info(

        f"Metrics saved to : {output_path}"

    )


# ==============================================================================
# Run Complete Evaluation
# ==============================================================================

def run_evaluation() -> tuple[float, float]:
    """
    Execute complete evaluation pipeline.

    Returns
    -------
    tuple
        (test_loss, test_accuracy)
    """

    (
        test_loss,
        test_accuracy,
        predictions,
        labels

    ) = evaluate_model()

    generate_classification_report(

        predictions,
        labels

    )

    generate_confusion_matrix(

        predictions,
        labels

    )

    save_prediction_results(

        predictions,
        labels

    )

    generate_evaluation_visualizations(

        predictions,
        labels

    )

    save_final_metrics(

        test_loss,
        test_accuracy

    )

    verify_evaluation(

        test_loss,
        test_accuracy

    )

    LOGGER.info("=" * 80)
    LOGGER.info("Evaluation Completed Successfully")
    LOGGER.info("=" * 80)

    return (

        test_loss,
        test_accuracy

    )


# ==============================================================================
# End of Part 6
# ==============================================================================

# End of Phase 5

# ==============================================================================
# ==============================================================================
# ==============================================================================
#
# ==============================================================================
#
# Phase 6
#
# Main Pipeline
#
# Part 1
#
# Execute Complete Pipeline
#
# ==============================================================================


def run_training_pipeline() -> None:
    """
    Execute the complete PlantXAI-Net training pipeline.

    Returns
    -------
    None
    """

    LOGGER.info("=" * 80)
    LOGGER.info("PlantXAI-Net Training Pipeline Started")
    LOGGER.info("=" * 80)

    # ----------------------------------------------------------
    # Verify Project Configuration
    # ----------------------------------------------------------

    verify_project_configuration()

    # ----------------------------------------------------------
    # Display System Information
    # ----------------------------------------------------------

    log_system_information()

    # ----------------------------------------------------------
    # Verify Dataset
    # ----------------------------------------------------------

    verify_all_dataloaders()

    display_dataset_summary()

    # ----------------------------------------------------------
    # Train Model
    # ----------------------------------------------------------

    LOGGER.info("=" * 80)
    LOGGER.info("Starting Model Training")
    LOGGER.info("=" * 80)

    train_model()

    # ----------------------------------------------------------
    # Evaluate Model
    # ----------------------------------------------------------

    LOGGER.info("=" * 80)
    LOGGER.info("Starting Model Evaluation")
    LOGGER.info("=" * 80)

    test_loss, test_accuracy = run_evaluation()

    # ----------------------------------------------------------
    # Display Final Summary
    # ----------------------------------------------------------

    display_project_summary(

        test_loss,

        test_accuracy

    )

    LOGGER.info("=" * 80)
    LOGGER.info("PlantXAI-Net Training Pipeline Finished Successfully")
    LOGGER.info("=" * 80)

# ==============================================================================
# End of Part 1
# ==============================================================================
#
# Phase 6
#
# Main Pipeline
#
# Part 2
#
# Verify Project Configuration
#
# ==============================================================================


def verify_project_configuration() -> None:
    """
    Verify project configuration before training.

    Returns
    -------
    None
    """

    LOGGER.info("=" * 80)

    LOGGER.info("Verifying Project Configuration")

    LOGGER.info("=" * 80)

    required_directories = [

        TRAIN_DIR,

        VALIDATION_DIR,

        TEST_DIR,

        CHECKPOINT_DIR,

        PLOT_DIR,

        REPORT_DIR

    ]

    for directory in required_directories:

        if not directory.exists():

            raise FileNotFoundError(

                f"Directory not found : {directory}"

            )

        LOGGER.info(

            f"Verified : {directory}"

        )

    LOGGER.info("=" * 80)

    LOGGER.info("Configuration Verification Successful")

    LOGGER.info("=" * 80)


# ==============================================================================
# End of Part 2
# ==============================================================================
# ==============================================================================
#
# Phase 6
#
# Main Pipeline
#
# Part 3
#
# Execute Pipeline Safely
#
# ==============================================================================


def execute_pipeline() -> None:
    """
    Execute complete pipeline safely.

    Returns
    -------
    None
    """

    execution_start_time = time.time()

    try:

        run_training_pipeline()

        execution_time = (

            time.time()

            - execution_start_time

        )

        LOGGER.info("=" * 80)

        LOGGER.info("Pipeline Finished Successfully")

        LOGGER.info("=" * 80)

        LOGGER.info(

            f"Execution Time : {execution_time:.2f} sec"

        )

        print()

        print("=" * 80)

        print("PLANTXAI-NET PIPELINE COMPLETED SUCCESSFULLY")

        print("=" * 80)

        print(

            f"Execution Time : {execution_time:.2f} sec"

        )

        print("=" * 80)

        print()

    except KeyboardInterrupt:

        LOGGER.warning(

            "Pipeline interrupted by user."

        )

        print()

        print("Training Interrupted.")

        print()

    except Exception as error:

        LOGGER.exception(

            "Pipeline execution failed."

        )

        print()

        print("=" * 80)

        print("PIPELINE FAILED")

        print("=" * 80)

        print(error)

        print("=" * 80)

        print()

        raise


# ==============================================================================
# End of Part 3
# ==============================================================================
# ==============================================================================
#
# Phase 6
#
# Main Pipeline
#
# Part 4
#
# Log System Information
#
# ==============================================================================

import platform


def log_system_information() -> None:
    """
    Display system information.

    Returns
    -------
    None
    """

    LOGGER.info("=" * 80)

    LOGGER.info("System Information")

    LOGGER.info("=" * 80)

    LOGGER.info(

        f"Operating System : {platform.system()}"

    )

    LOGGER.info(

        f"OS Version : {platform.release()}"

    )

    LOGGER.info(

        f"Python Version : {platform.python_version()}"

    )

    LOGGER.info(

        f"PyTorch Version : {torch.__version__}"

    )

    LOGGER.info(

        f"Device : {DEVICE}"

    )

    LOGGER.info(

        f"CUDA Available : {torch.cuda.is_available()}"

    )

    if torch.cuda.is_available():

        LOGGER.info(

            f"GPU Name : {torch.cuda.get_device_name(0)}"

        )

        LOGGER.info(

            f"GPU Count : {torch.cuda.device_count()}"

        )

        LOGGER.info(

            f"CUDA Version : {torch.version.cuda}"

        )

    LOGGER.info("=" * 80)

    print()

    print("=" * 80)

    print("SYSTEM INFORMATION")

    print("=" * 80)

    print(

        f"Operating System : {platform.system()}"

    )

    print(

        f"Python Version : {platform.python_version()}"

    )

    print(

        f"PyTorch Version : {torch.__version__}"

    )

    print(

        f"Device : {DEVICE}"

    )

    if torch.cuda.is_available():

        print(

            f"GPU : {torch.cuda.get_device_name(0)}"

        )

    else:

        print(

            "GPU : Not Available"

        )

    print("=" * 80)

    print()


# ==============================================================================
# End of Part 4
# ==============================================================================
# ==============================================================================
# ==============================================================================
# ==============================================================================
#
# Phase 6
#
# Part 5
#
# Final Project Summary
#
# ==============================================================================


def display_project_summary(
    test_loss: float,
    test_accuracy: float
) -> None:
    """
    Display final project summary.

    Parameters
    ----------
    test_loss : float

    test_accuracy : float

    Returns
    -------
    None
    """

    LOGGER.info("=" * 80)
    LOGGER.info("PlantXAI-Net Project Summary")
    LOGGER.info("=" * 80)

    LOGGER.info(f"Model Architecture      : {MODEL.__class__.__name__}")
    LOGGER.info(f"Total Classes           : {NUM_CLASSES}")
    LOGGER.info(f"Training Epochs         : {NUM_EPOCHS}")
    LOGGER.info(f"Batch Size              : {BATCH_SIZE}")
    LOGGER.info(f"Learning Rate           : {LEARNING_RATE}")
    LOGGER.info(f"Device                  : {DEVICE}")

    LOGGER.info("-" * 80)

    LOGGER.info(f"Best Validation Accuracy : {BEST_VALIDATION_ACCURACY:.4f}")
    LOGGER.info(f"Best Validation Loss     : {BEST_VALIDATION_LOSS:.4f}")

    LOGGER.info("-" * 80)

    LOGGER.info(f"Final Test Accuracy      : {test_accuracy:.4f}")
    LOGGER.info(f"Final Test Loss          : {test_loss:.4f}")

    LOGGER.info("-" * 80)

    LOGGER.info("Generated Files")

    LOGGER.info(f"Checkpoints : {CHECKPOINT_DIR}")
    LOGGER.info(f"Plots       : {PLOT_DIR}")
    LOGGER.info(f"Reports     : {REPORT_DIR}")

    LOGGER.info("=" * 80)

    print()

    print("=" * 80)
    print("PLANTXAI-NET PROJECT SUMMARY")
    print("=" * 80)

    print(f"Model Architecture : {MODEL.__class__.__name__}")
    print(f"Total Classes      : {NUM_CLASSES}")
    print(f"Epochs             : {NUM_EPOCHS}")
    print(f"Batch Size         : {BATCH_SIZE}")
    print(f"Learning Rate      : {LEARNING_RATE}")
    print(f"Device             : {DEVICE}")

    print("-" * 80)

    print(f"Best Validation Accuracy : {BEST_VALIDATION_ACCURACY:.4f}")
    print(f"Best Validation Loss     : {BEST_VALIDATION_LOSS:.4f}")

    print("-" * 80)

    print(f"Final Test Accuracy      : {test_accuracy:.4f}")
    print(f"Final Test Loss          : {test_loss:.4f}")

    print("-" * 80)

    print("Generated Files")

    print(f"Checkpoints : {CHECKPOINT_DIR}")
    print(f"Plots       : {PLOT_DIR}")
    print(f"Reports     : {REPORT_DIR}")

    print("=" * 80)
    print()


# ==============================================================================
# End of Part 5
# ==============================================================================
#
# Phase 6
#
# Part 6
#
# Program Entry Point
#
# ==============================================================================


def main() -> None:
    """
    Main entry point for PlantXAI-Net.

    Returns
    -------
    None
    """

    LOGGER.info("=" * 80)
    LOGGER.info("PlantXAI-Net Started")
    LOGGER.info("=" * 80)

    execute_pipeline()

    LOGGER.info("=" * 80)
    LOGGER.info("PlantXAI-Net Finished")
    LOGGER.info("=" * 80)


# ==============================================================================
# Program Entry
# ==============================================================================
if __name__ == "__main__":

    verify_existing_checkpoint()