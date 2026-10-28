# ==============================================================================
# PlantXAI-Net
#
# Baseline Model: ResNet-50
#
# Purpose:
#   ResNet-50 is used as a baseline comparison model against EfficientNet-B0.
#
# Training Strategy:
#   ImageNet-pretrained ResNet-50
#   Head-only transfer learning
#
# Important:
#   Loss and accuracy are displayed ONLY after each epoch.
#   tqdm displays ONLY live batch progress.
# ==============================================================================


# ==============================================================================
# Imports
# ==============================================================================

import time
import random
import warnings

from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import transforms
from torchvision import models

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from tqdm import tqdm

import matplotlib.pyplot as plt


warnings.filterwarnings("ignore")


# ==============================================================================
# Project Configuration
# ==============================================================================

PROJECT_NAME = "PlantXAI-Net"

MODEL_NAME = "ResNet-50"

IMAGE_SIZE = 224

BATCH_SIZE = 32

NUM_EPOCHS = 8

LEARNING_RATE = 0.001

WEIGHT_DECAY = 1e-4

NUM_WORKERS = 0

PIN_MEMORY = True

RANDOM_SEED = 42

SHUFFLE = True


# ==============================================================================
# Project Directories
# ==============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


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


BASELINE_MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "baselines"
)


BASELINE_CHECKPOINT_DIR = (
    BASELINE_MODEL_DIR
    / "checkpoints"
)


RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "baselines"
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
# ResNet-50 Local ImageNet Weight
# ==============================================================================
#
# IMPORTANT:
# Put the downloaded ResNet-50 ImageNet .pth file at:
#
# D:\PlantXAI-Net\models\baselines\resnet50-11ad3fa6.pth
#
# If your file has another name, change only this variable.
#
# ==============================================================================

RESNET_WEIGHTS_PATH = (
    BASELINE_MODEL_DIR
    / "resnet50-11ad3fa6.pth"
)


# ==============================================================================
# Output Files
# ==============================================================================

BEST_MODEL_PATH = (
    BASELINE_CHECKPOINT_DIR
    / "resnet50_best_model.pth"
)


LAST_CHECKPOINT_PATH = (
    BASELINE_CHECKPOINT_DIR
    / "resnet50_last_checkpoint.pth"
)


TRAINING_HISTORY_PATH = (
    METRIC_DIR
    / "resnet50_training_history.csv"
)


TEST_RESULTS_PATH = (
    METRIC_DIR
    / "resnet50_test_results.csv"
)


CLASSIFICATION_REPORT_PATH = (
    REPORT_DIR
    / "resnet50_classification_report.txt"
)


CONFUSION_MATRIX_PATH = (
    REPORT_DIR
    / "resnet50_confusion_matrix.csv"
)


# ==============================================================================
# Device
# ==============================================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ==============================================================================
# Random Seed
# ==============================================================================

def set_random_seed(seed=RANDOM_SEED):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True

        torch.backends.cudnn.benchmark = False


# ==============================================================================
# Create Directories
# ==============================================================================

def create_directories():

    BASELINE_CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    RESULTS_DIR.mkdir(
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
# Dataset Validation
# ==============================================================================

def validate_dataset():

    required_directories = {

        "Training Dataset": TRAIN_DIR,

        "Validation Dataset": VALIDATION_DIR,

        "Testing Dataset": TEST_DIR

    }


    for name, directory in required_directories.items():

        if not directory.exists():

            raise FileNotFoundError(
                f"{name} not found:\n{directory}"
            )


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
# Training Transform
# ==============================================================================
#
# Same augmentation philosophy as the EfficientNet training pipeline.
#
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
# Validation Transform
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
# Test Transform
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
# Load Datasets
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

NUM_CLASSES = len(CLASS_NAMES)


TRAIN_IMAGES = len(TRAIN_DATASET)

VALIDATION_IMAGES = len(VALIDATION_DATASET)

TEST_IMAGES = len(TEST_DATASET)


# ==============================================================================
# Dataset Consistency
# ==============================================================================

if TRAIN_DATASET.classes != VALIDATION_DATASET.classes:

    raise RuntimeError(
        "Training and validation classes do not match."
    )


if TRAIN_DATASET.classes != TEST_DATASET.classes:

    raise RuntimeError(
        "Training and testing classes do not match."
    )


# ==============================================================================
# DataLoaders
# ==============================================================================

TRAIN_LOADER = DataLoader(
    TRAIN_DATASET,
    batch_size=BATCH_SIZE,
    shuffle=SHUFFLE,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    drop_last=False
)


VALIDATION_LOADER = DataLoader(
    VALIDATION_DATASET,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    drop_last=False
)


TEST_LOADER = DataLoader(
    TEST_DATASET,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    drop_last=False
)


# ==============================================================================
# Load ResNet-50 Architecture
# ==============================================================================

print()
print("=" * 80)
print("PlantXAI-Net")
print("Baseline Model: ResNet-50")
print("=" * 80)

print()

print(f"Project Root : {PROJECT_ROOT}")

print(f"Train Dir   : {TRAIN_DIR}")

print(f"Val Dir     : {VALIDATION_DIR}")

print(f"Test Dir    : {TEST_DIR}")

print()

print(f"Device      : {DEVICE}")

print()

print(f"Number of Classes : {NUM_CLASSES}")

print()

print("Dataset Statistics")
print("-" * 80)

print(
    f"Training Images   : {TRAIN_IMAGES}"
)

print(
    f"Validation Images : {VALIDATION_IMAGES}"
)

print(
    f"Test Images       : {TEST_IMAGES}"
)

print()

print("Dataset verification: PASSED")

print()


# ==============================================================================
# Load ImageNet Pretrained Weights
# ==============================================================================

print("=" * 80)
print("LOADING IMAGENET-PRETRAINED RESNET-50")
print("=" * 80)

print()


if not RESNET_WEIGHTS_PATH.exists():

    raise FileNotFoundError(
        "\nResNet-50 ImageNet weights were not found.\n\n"
        f"Expected location:\n{RESNET_WEIGHTS_PATH}\n\n"
        "Please place the ResNet-50 .pth weight file there."
    )


try:

    model = models.resnet50(
        weights=None
    )

    checkpoint_weights = torch.load(
        RESNET_WEIGHTS_PATH,
        map_location="cpu"
    )


    # --------------------------------------------------------------------------
    # Handle different checkpoint formats
    # --------------------------------------------------------------------------

    if isinstance(checkpoint_weights, dict):

        if "state_dict" in checkpoint_weights:

            checkpoint_weights = (
                checkpoint_weights["state_dict"]
            )

        elif "model_state_dict" in checkpoint_weights:

            checkpoint_weights = (
                checkpoint_weights["model_state_dict"]
            )


    # --------------------------------------------------------------------------
    # Remove possible "module." prefix
    # --------------------------------------------------------------------------

    cleaned_weights = {}

    for key, value in checkpoint_weights.items():

        if key.startswith("module."):

            key = key.replace(
                "module.",
                "",
                1
            )

        cleaned_weights[key] = value


    model.load_state_dict(
        cleaned_weights,
        strict=True
    )


    print(
        "ImageNet pretrained weights loaded successfully."
    )


except Exception as error:

    raise RuntimeError(
        "\nFailed to load the local ResNet-50 ImageNet weights.\n\n"
        f"Weight file:\n{RESNET_WEIGHTS_PATH}\n\n"
        f"Reason:\n{error}"
    )


# ==============================================================================
# Freeze ResNet-50 Backbone
# ==============================================================================

for parameter in model.parameters():

    parameter.requires_grad = False


# ==============================================================================
# Replace Classification Head
# ==============================================================================

IN_FEATURES = model.fc.in_features


model.fc = nn.Linear(
    IN_FEATURES,
    NUM_CLASSES
)


# ==============================================================================
# Enable Training of Classification Head
# ==============================================================================

for parameter in model.fc.parameters():

    parameter.requires_grad = True


# ==============================================================================
# Move Model to Device
# ==============================================================================

model = model.to(DEVICE)


# ==============================================================================
# Parameter Statistics
# ==============================================================================

TOTAL_PARAMETERS = sum(
    parameter.numel()
    for parameter in model.parameters()
)


TRAINABLE_PARAMETERS = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)


FROZEN_PARAMETERS = (
    TOTAL_PARAMETERS
    - TRAINABLE_PARAMETERS
)


# ==============================================================================
# Model Information
# ==============================================================================

print()

print("Model Information")
print("-" * 80)

print(
    f"Architecture          : {MODEL_NAME}"
)

print(
    "Pretrained Weights     : ImageNet"
)

print(
    "Transfer Learning      : Head-only"
)

print(
    f"Total Parameters      : {TOTAL_PARAMETERS:,}"
)

print(
    f"Trainable Parameters  : {TRAINABLE_PARAMETERS:,}"
)

print(
    f"Frozen Parameters     : {FROZEN_PARAMETERS:,}"
)

print()


# ==============================================================================
# Loss Function
# ==============================================================================

CRITERION = nn.CrossEntropyLoss()


# ==============================================================================
# Optimizer
# ==============================================================================

OPTIMIZER = optim.AdamW(
    filter(
        lambda parameter:
        parameter.requires_grad,
        model.parameters()
    ),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ==============================================================================
# Learning Rate Scheduler
# ==============================================================================

SCHEDULER = optim.lr_scheduler.ReduceLROnPlateau(
    OPTIMIZER,
    mode="min",
    factor=0.1,
    patience=3,
    threshold=1e-4,
    min_lr=1e-6
)


# ==============================================================================
# Training History
# ==============================================================================

TRAIN_LOSS_HISTORY = []

TRAIN_ACCURACY_HISTORY = []

VALIDATION_LOSS_HISTORY = []

VALIDATION_ACCURACY_HISTORY = []


BEST_VALIDATION_ACCURACY = 0.0

BEST_VALIDATION_LOSS = float("inf")


# ==============================================================================
# Train One Epoch
# ==============================================================================

def train_one_epoch(
    epoch
):

    model.train()

    running_loss = 0.0

    correct_predictions = 0

    total_samples = 0


    # --------------------------------------------------------------------------
    # tqdm ONLY
    #
    # No set_postfix()
    # No live loss
    # No live accuracy
    #
    # --------------------------------------------------------------------------

    progress_bar = tqdm(
        TRAIN_LOADER,
        desc=f"Training Epoch {epoch + 1}/{NUM_EPOCHS}",
        leave=False,
        dynamic_ncols=True
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


        outputs = model(
            images
        )


        loss = CRITERION(
            outputs,
            labels
        )


        loss.backward()

        OPTIMIZER.step()


        predictions = torch.argmax(
            outputs,
            dim=1
        )


        batch_size = labels.size(0)


        running_loss += (
            loss.item()
            * batch_size
        )


        correct_predictions += (
            predictions == labels
        ).sum().item()


        total_samples += batch_size


    epoch_loss = (
        running_loss
        /
        total_samples
    )


    epoch_accuracy = (
        correct_predictions
        /
        total_samples
    )


    return (
        epoch_loss,
        epoch_accuracy
    )


# ==============================================================================
# Validate One Epoch
# ==============================================================================

def validate_one_epoch(
    epoch
):

    model.eval()

    running_loss = 0.0

    correct_predictions = 0

    total_samples = 0


    progress_bar = tqdm(
        VALIDATION_LOADER,
        desc=f"Validation Epoch {epoch + 1}/{NUM_EPOCHS}",
        leave=False,
        dynamic_ncols=True
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


            outputs = model(
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


            batch_size = labels.size(0)


            running_loss += (
                loss.item()
                * batch_size
            )


            correct_predictions += (
                predictions == labels
            ).sum().item()


            total_samples += batch_size


    validation_loss = (
        running_loss
        /
        total_samples
    )


    validation_accuracy = (
        correct_predictions
        /
        total_samples
    )


    return (
        validation_loss,
        validation_accuracy
    )


# ==============================================================================
# Save Checkpoint
# ==============================================================================

def save_checkpoint(
    epoch,
    train_loss,
    train_accuracy,
    validation_loss,
    validation_accuracy
):

    checkpoint = {

        "epoch": epoch,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            OPTIMIZER.state_dict(),

        "scheduler_state_dict":
            SCHEDULER.state_dict(),

        "train_loss":
            train_loss,

        "train_accuracy":
            train_accuracy,

        "validation_loss":
            validation_loss,

        "validation_accuracy":
            validation_accuracy,

        "train_loss_history":
            TRAIN_LOSS_HISTORY,

        "train_accuracy_history":
            TRAIN_ACCURACY_HISTORY,

        "validation_loss_history":
            VALIDATION_LOSS_HISTORY,

        "validation_accuracy_history":
            VALIDATION_ACCURACY_HISTORY

    }


    torch.save(
        checkpoint,
        LAST_CHECKPOINT_PATH
    )


    if (
        validation_accuracy
        >
        BEST_VALIDATION_ACCURACY
    ):

        torch.save(
            checkpoint,
            BEST_MODEL_PATH
        )


# ==============================================================================
# Train Model
# ==============================================================================

def train_model():

    global BEST_VALIDATION_ACCURACY

    global BEST_VALIDATION_LOSS


    print("=" * 80)

    print("RESNET-50 BASELINE TRAINING")

    print("=" * 80)

    print()


    for epoch in range(
        NUM_EPOCHS
    ):


        epoch_start_time = time.time()


        # ======================================================================
        # Training
        # ======================================================================

        train_loss, train_accuracy = (
            train_one_epoch(
                epoch
            )
        )


        # ======================================================================
        # Validation
        # ======================================================================

        validation_loss, validation_accuracy = (
            validate_one_epoch(
                epoch
            )
        )


        # ======================================================================
        # Learning Rate Scheduler
        # ======================================================================

        SCHEDULER.step(
            validation_loss
        )


        # ======================================================================
        # Store History
        # ======================================================================

        TRAIN_LOSS_HISTORY.append(
            train_loss
        )

        TRAIN_ACCURACY_HISTORY.append(
            train_accuracy
        )

        VALIDATION_LOSS_HISTORY.append(
            validation_loss
        )

        VALIDATION_ACCURACY_HISTORY.append(
            validation_accuracy
        )


        # ======================================================================
        # Best Model
        # ======================================================================

        is_best = (

            validation_accuracy
            >
            BEST_VALIDATION_ACCURACY

        ) or (

            validation_accuracy
            ==
            BEST_VALIDATION_ACCURACY

            and

            validation_loss
            <
            BEST_VALIDATION_LOSS

        )


        if is_best:

            BEST_VALIDATION_ACCURACY = (
                validation_accuracy
            )

            BEST_VALIDATION_LOSS = (
                validation_loss
            )


        # ======================================================================
        # Save Checkpoint
        # ======================================================================

        save_checkpoint(
            epoch,
            train_loss,
            train_accuracy,
            validation_loss,
            validation_accuracy
        )


        # ======================================================================
        # Epoch Time
        # ======================================================================

        epoch_time = (
            time.time()
            -
            epoch_start_time
        )


        # ======================================================================
        # IMPORTANT:
        # Loss and Accuracy are printed ONLY HERE.
        # ======================================================================

        print()

        print("-" * 80)

        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} Completed"
        )

        print(
            f"Train Loss          : {train_loss:.4f}"
        )

        print(
            f"Train Accuracy      : "
            f"{train_accuracy * 100:.2f}%"
        )

        print(
            f"Validation Loss     : "
            f"{validation_loss:.4f}"
        )

        print(
            f"Validation Accuracy : "
            f"{validation_accuracy * 100:.2f}%"
        )

        print(
            f"Learning Rate       : "
            f"{OPTIMIZER.param_groups[0]['lr']:.6f}"
        )

        print(
            f"Epoch Time          : "
            f"{epoch_time / 60:.2f} min"
        )

        if is_best:

            print(
                "Status              : NEW BEST MODEL"
            )

        else:

            print(
                "Status              : No improvement"
            )

        print("-" * 80)


    print()

    print("=" * 80)

    print("RESNET-50 TRAINING COMPLETED")

    print("=" * 80)

    print(
        f"Epochs Completed       : {len(TRAIN_LOSS_HISTORY)}"
    )

    print(
        f"Best Validation Acc.   : "
        f"{BEST_VALIDATION_ACCURACY * 100:.2f}%"
    )

    print(
        f"Best Validation Loss   : "
        f"{BEST_VALIDATION_LOSS:.4f}"
    )

    print(
        f"Best Model             : {BEST_MODEL_PATH}"
    )

    print("=" * 80)


# ==============================================================================
# Save Training History
# ==============================================================================

def save_training_history():

    history = pd.DataFrame(
        {

            "Epoch":
                list(
                    range(
                        1,
                        len(
                            TRAIN_LOSS_HISTORY
                        ) + 1
                    )
                ),

            "Train Loss":
                TRAIN_LOSS_HISTORY,

            "Validation Loss":
                VALIDATION_LOSS_HISTORY,

            "Train Accuracy":
                TRAIN_ACCURACY_HISTORY,

            "Validation Accuracy":
                VALIDATION_ACCURACY_HISTORY

        }
    )


    history.to_csv(
        TRAINING_HISTORY_PATH,
        index=False
    )


    # --------------------------------------------------------------------------
    # Loss Curve
    # --------------------------------------------------------------------------

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

    plt.title(
        "ResNet-50 Baseline Loss Curve"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()


    plt.savefig(
        PLOT_DIR / "resnet50_loss_curve.png",
        dpi=300
    )


    plt.close()


    # --------------------------------------------------------------------------
    # Accuracy Curve
    # --------------------------------------------------------------------------

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

    plt.title(
        "ResNet-50 Baseline Accuracy Curve"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()


    plt.savefig(
        PLOT_DIR / "resnet50_accuracy_curve.png",
        dpi=300
    )


    plt.close()


    print()

    print(
        f"Training history saved : "
        f"{TRAINING_HISTORY_PATH}"
    )

    print(
        f"Plots saved            : "
        f"{PLOT_DIR}"
    )


# ==============================================================================
# Test Evaluation
# ==============================================================================

def evaluate_test_set():

    print()

    print("=" * 80)

    print("RESNET-50 BASELINE TEST EVALUATION")

    print("=" * 80)

    print()


    model.eval()


    running_loss = 0.0

    all_predictions = []

    all_labels = []


    progress_bar = tqdm(
        TEST_LOADER,
        desc="Testing",
        leave=False,
        dynamic_ncols=True
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


            outputs = model(
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


            batch_size = labels.size(0)


            running_loss += (
                loss.item()
                * batch_size
            )


            all_predictions.extend(
                predictions.cpu().tolist()
            )


            all_labels.extend(
                labels.cpu().tolist()
            )


    test_loss = (
        running_loss
        /
        len(TEST_DATASET)
    )


    test_accuracy = accuracy_score(
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


    # ==========================================================================
    # Display Final Test Results
    # ==========================================================================

    print()

    print("-" * 80)

    print("RESNET-50 TEST RESULTS")

    print("-" * 80)

    print(
        f"Test Images       : {len(TEST_DATASET)}"
    )

    print(
        f"Test Loss         : {test_loss:.4f}"
    )

    print(
        f"Test Accuracy     : "
        f"{test_accuracy * 100:.2f}%"
    )

    print(
        f"Weighted Precision: "
        f"{precision * 100:.2f}%"
    )

    print(
        f"Weighted Recall   : "
        f"{recall * 100:.2f}%"
    )

    print(
        f"Weighted F1 Score : "
        f"{f1 * 100:.2f}%"
    )

    print("-" * 80)


    # ==========================================================================
    # Save Test Metrics
    # ==========================================================================

    results = pd.DataFrame(
        [

            {

                "Model":
                    MODEL_NAME,

                "Test Images":
                    len(TEST_DATASET),

                "Test Loss":
                    test_loss,

                "Test Accuracy":
                    test_accuracy,

                "Weighted Precision":
                    precision,

                "Weighted Recall":
                    recall,

                "Weighted F1":
                    f1

            }

        ]
    )


    results.to_csv(
        TEST_RESULTS_PATH,
        index=False
    )


    # ==========================================================================
    # Classification Report
    # ==========================================================================

    report = classification_report(
        all_labels,
        all_predictions,
        target_names=CLASS_NAMES,
        zero_division=0
    )


    with open(
        CLASSIFICATION_REPORT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            report
        )


    # ==========================================================================
    # Confusion Matrix
    # ==========================================================================

    matrix = confusion_matrix(
        all_labels,
        all_predictions
    )


    matrix_dataframe = pd.DataFrame(
        matrix,
        index=CLASS_NAMES,
        columns=CLASS_NAMES
    )


    matrix_dataframe.to_csv(
        CONFUSION_MATRIX_PATH
    )


    print()

    print(
        f"Test metrics saved : "
        f"{TEST_RESULTS_PATH}"
    )

    print(
        f"Classification report saved : "
        f"{CLASSIFICATION_REPORT_PATH}"
    )

    print(
        f"Confusion matrix saved : "
        f"{CONFUSION_MATRIX_PATH}"
    )


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":

    set_random_seed()

    create_directories()

    validate_dataset()

    train_model()

    save_training_history()

    evaluate_test_set()

    print()

    print("=" * 80)

    print("RESNET-50 BASELINE PIPELINE COMPLETED SUCCESSFULLY")

    print("=" * 80)

    print()