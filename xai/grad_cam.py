# ==============================================================================
#
# PlantXAI-Net
#
# Phase 5
#
# XAI / Explainable AI
#
# Part 1
#
# Grad-CAM Configuration + Model Loading
#
# ==============================================================================


# ==============================================================================
# Imports
# ==============================================================================

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from torchvision import models
from torchvision import datasets


# ==============================================================================
# Project Configuration
# ==============================================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)


# ==============================================================================
# Model Configuration
# ==============================================================================

MODEL_NAME = "EfficientNet-B0"

IMAGE_SIZE = 224

NUM_CLASSES = 38


# ==============================================================================
# Dataset Configuration
# ==============================================================================

DATASET_ROOT = (
    PROJECT_ROOT
    / "dataset"
    / "PlantVillage"
)


TRAIN_DIR = (
    DATASET_ROOT
    / "balanced"
)


TEST_DIR = (
    DATASET_ROOT
    / "split"
    / "test"
)


# ==============================================================================
# Checkpoint Configuration
# ==============================================================================

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "models"
    / "checkpoints"
)


# IMPORTANT:
# Use the same Epoch-8 checkpoint that produced the verified
# 95.83% test accuracy on the 5,448-image test set.

LAST_CHECKPOINT_PATH = (
    CHECKPOINT_DIR
    / "last_checkpoint.pth"
)


# ==============================================================================
# XAI Results Configuration
# ==============================================================================

XAI_RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "xai"
)


XAI_RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==============================================================================
# Device Configuration
# ==============================================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
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
# Load Class Names
# ==============================================================================

def load_class_names() -> list[str]:

    """
    Load class names using the same dataset structure
    used during model training.

    Returns
    -------
    list[str]
        Ordered list of class names.
    """

    if not TRAIN_DIR.exists():

        raise FileNotFoundError(
            f"Training dataset directory not found: "
            f"{TRAIN_DIR}"
        )


    dataset = datasets.ImageFolder(
        root=TRAIN_DIR
    )


    class_names = dataset.classes


    if len(class_names) != NUM_CLASSES:

        raise RuntimeError(
            "Number of dataset classes does not match "
            f"expected number of classes. "
            f"Expected: {NUM_CLASSES}, "
            f"Found: {len(class_names)}"
        )


    return class_names


# ==============================================================================
# Build EfficientNet-B0
# ==============================================================================

def build_model() -> nn.Module:

    """
    Build the same EfficientNet-B0 architecture
    used during model training.
    """

    model = models.efficientnet_b0(
        weights=None
    )


    # --------------------------------------------------------------------------
    # Replace Classification Head
    # --------------------------------------------------------------------------

    in_features = (
        model.classifier[1].in_features
    )


    model.classifier = nn.Sequential(

        nn.Dropout(
            p=0.30
        ),

        nn.Linear(
            in_features=in_features,
            out_features=NUM_CLASSES
        )

    )


    return model


# ==============================================================================
# Load Trained Model
# ==============================================================================

def load_trained_model() -> nn.Module:

    """
    Load the Epoch-8 trained EfficientNet-B0 checkpoint
    used for the verified 95.83% test accuracy.
    """

    if not LAST_CHECKPOINT_PATH.exists():

        raise FileNotFoundError(
            f"Last checkpoint not found: "
            f"{LAST_CHECKPOINT_PATH}"
        )


    model = build_model()


    checkpoint = torch.load(
        LAST_CHECKPOINT_PATH,
        map_location=DEVICE
    )


    # --------------------------------------------------------------------------
    # Validate Checkpoint
    # --------------------------------------------------------------------------

    if "model_state_dict" not in checkpoint:

        raise KeyError(
            "model_state_dict not found in "
            "last_checkpoint.pth."
        )


    # --------------------------------------------------------------------------
    # Load Model State
    # --------------------------------------------------------------------------

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )


    # --------------------------------------------------------------------------
    # Move Model to Device
    # --------------------------------------------------------------------------

    model = model.to(
        DEVICE
    )


    model.eval()


    return model


# ==============================================================================
# Select Grad-CAM Target Layer
# ==============================================================================

def get_target_layer(
    model: nn.Module
) -> nn.Module:

    """
    Select the final convolutional feature block
    of EfficientNet-B0 for Grad-CAM.
    """

    target_layer = (
        model.features[-1]
    )


    return target_layer


# ==============================================================================
# Verify Grad-CAM Configuration
# ==============================================================================

def verify_gradcam_setup() -> None:

    """
    Verify model, classes, checkpoint and target layer.
    """

    print()

    print("=" * 80)

    print(
        "PlantXAI-Net"
    )

    print(
        "Phase 5 - Grad-CAM Setup Verification"
    )

    print("=" * 80)

    print()


    print(
        f"Project Root       : {PROJECT_ROOT}"
    )

    print(
        f"Model              : {MODEL_NAME}"
    )

    print(
        f"Device             : {DEVICE}"
    )

    print(
        f"Image Size         : "
        f"{IMAGE_SIZE} x {IMAGE_SIZE}"
    )

    print(
        f"Expected Classes   : {NUM_CLASSES}"
    )

    print()


    print(
        f"Checkpoint         : "
        f"{LAST_CHECKPOINT_PATH}"
    )

    print(
        f"Checkpoint Exists  : "
        f"{LAST_CHECKPOINT_PATH.exists()}"
    )

    print()


    # --------------------------------------------------------------------------
    # Load Classes
    # --------------------------------------------------------------------------

    class_names = load_class_names()


    print(
        f"Classes Loaded     : "
        f"{len(class_names)}"
    )

    print()


    # --------------------------------------------------------------------------
    # Load Trained Model
    # --------------------------------------------------------------------------

    model = load_trained_model()


    print(
        f"Model Loaded       : "
        f"{MODEL_NAME}"
    )

    print(
        f"Model Device       : "
        f"{next(model.parameters()).device}"
    )

    print()


    # --------------------------------------------------------------------------
    # Display Checkpoint Information
    # --------------------------------------------------------------------------

    checkpoint = torch.load(
        LAST_CHECKPOINT_PATH,
        map_location="cpu"
    )


    if "epoch" in checkpoint:

        print(
            f"Checkpoint Epoch   : "
            f"{checkpoint['epoch']}"
        )

        print(
            f"Training Epoch     : "
            f"{checkpoint['epoch'] + 1}"
        )


    if "validation_accuracy" in checkpoint:

        print(
            f"Validation Accuracy: "
            f"{checkpoint['validation_accuracy'] * 100:.4f}%"
        )


    if "validation_loss" in checkpoint:

        print(
            f"Validation Loss    : "
            f"{checkpoint['validation_loss']:.6f}"
        )


    print()


    # --------------------------------------------------------------------------
    # Select Target Layer
    # --------------------------------------------------------------------------

    target_layer = get_target_layer(
        model
    )


    print(
        f"Grad-CAM Layer     : "
        f"{target_layer.__class__.__name__}"
    )

    print()


    # --------------------------------------------------------------------------
    # Forward Pass Verification
    # --------------------------------------------------------------------------

    dummy_input = torch.randn(
        1,
        3,
        IMAGE_SIZE,
        IMAGE_SIZE,
        device=DEVICE
    )


    with torch.no_grad():

        output = model(
            dummy_input
        )


    print(
        f"Model Output Shape : "
        f"{tuple(output.shape)}"
    )


    if output.shape[1] != NUM_CLASSES:

        raise RuntimeError(
            "Model output does not match the expected "
            f"{NUM_CLASSES} classes."
        )


    print()

    print(
        "Grad-CAM Setup     : SUCCESS"
    )

    print()

    print("=" * 80)

    print(
        "Phase 5 - Part 1 Completed Successfully"
    )

    print("=" * 80)

    print()


# ==============================================================================
# End of Part 1
# ==============================================================================

# Phase 5
#
# XAI / Explainable AI
#
# Part 2
#
# Image Prediction
#
# ==============================================================================


# ==============================================================================
# Additional Imports
# ==============================================================================

import sys

from PIL import Image

from torchvision import transforms


# ==============================================================================
# Image Transformation
# ==============================================================================

INFERENCE_TRANSFORM = transforms.Compose(

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
# Predict Image
# ==============================================================================

def predict_image(

    model: nn.Module,

    image_path: Path,

    class_names: list[str]

) -> tuple[str, float, int]:

    """
    Predict the disease class for a single plant image.

    Parameters
    ----------
    model : nn.Module
        Trained EfficientNet-B0 model.

    image_path : Path
        Path to input plant image.

    class_names : list[str]
        Ordered list of class names.

    Returns
    -------
    tuple
        Predicted class name,
        confidence score,
        predicted class index.
    """

    if not image_path.exists():

        raise FileNotFoundError(

            f"Image not found: {image_path}"

        )


    # --------------------------------------------------------------------------
    # Load Image
    # --------------------------------------------------------------------------

    image = Image.open(

        image_path

    ).convert(

        "RGB"

    )


    # --------------------------------------------------------------------------
    # Preprocess Image
    # --------------------------------------------------------------------------

    input_tensor = INFERENCE_TRANSFORM(

        image

    )


    input_tensor = input_tensor.unsqueeze(

        0

    ).to(

        DEVICE

    )


    # --------------------------------------------------------------------------
    # Model Prediction
    # --------------------------------------------------------------------------

    model.eval()


    with torch.no_grad():

        outputs = model(

            input_tensor

        )

        probabilities = torch.softmax(

            outputs,

            dim=1

        )


        confidence, predicted_index = torch.max(

            probabilities,

            dim=1

        )


    predicted_index = int(

        predicted_index.item()

    )


    confidence = float(

        confidence.item()

    )


    predicted_class = class_names[

        predicted_index

    ]


    return (

        predicted_class,

        confidence,

        predicted_index

    )


# ==============================================================================
# Display Prediction
# ==============================================================================

def display_prediction(

    image_path: Path,

    predicted_class: str,

    confidence: float

) -> None:

    """
    Display prediction information.

    Parameters
    ----------
    image_path : Path
        Input image path.

    predicted_class : str
        Predicted disease class.

    confidence : float
        Prediction confidence.

    Returns
    -------
    None
    """

    print()

    print("=" * 80)

    print("PLANT DISEASE PREDICTION")

    print("=" * 80)

    print()

    print(

        f"Input Image       : {image_path.name}"

    )

    print()

    print(

        f"Predicted Disease : {predicted_class}"

    )

    print()

    print(

        f"Confidence        : {confidence * 100:.2f}%"

    )

    print()

    print("=" * 80)


# ==============================================================================
# Get Image Path From Command Line
# ==============================================================================

def get_input_image_path() -> Path:

    """
    Get input image path from command-line arguments.

    Returns
    -------
    Path
        Input image path.
    """

    if len(sys.argv) < 2:

        raise ValueError(

            "Please provide an image path.\n\n"

            "Example:\n"

            "python xai\\grad_cam.py "
            "\"D:\\PlantXAI-Net\\test_image.jpg\""

        )


    image_path = Path(

        sys.argv[1]

    )


    return image_path




# ==============================================================================
# End of Phase 5 - Part 2
# ==============================================================================
# ==============================================================================
# ==============================================================================
#
# PlantXAI-Net
#
# Phase 5
#
# XAI / Explainable AI
#
# Part 3
#
# Grad-CAM Generation
#
# ==============================================================================


# ==============================================================================
# Image Preprocessing for Grad-CAM
# ==============================================================================

def preprocess_image(
    image_path: Path
) -> tuple[torch.Tensor, np.ndarray]:

    """
    Preprocess an image using the same inference transform
    used by the model prediction pipeline.

    Returns
    -------
    tuple
        input_tensor:
            Normalized tensor for model inference.

        rgb_image:
            Original RGB image used for visualization
            and heatmap resizing.
    """

    # --------------------------------------------------------------------------
    # Load Image Using PIL
    # --------------------------------------------------------------------------

    image = Image.open(
        image_path
    ).convert(
        "RGB"
    )


    # --------------------------------------------------------------------------
    # Keep Original RGB Image
    # --------------------------------------------------------------------------

    rgb_image = np.array(
        image
    )


    # --------------------------------------------------------------------------
    # Apply SAME Inference Transform Used in Part 2
    # --------------------------------------------------------------------------

    input_tensor = INFERENCE_TRANSFORM(
        image
    )


    # --------------------------------------------------------------------------
    # Add Batch Dimension
    # --------------------------------------------------------------------------

    input_tensor = input_tensor.unsqueeze(
        0
    )


    return (
        input_tensor,
        rgb_image
    )


# ==============================================================================
# Generate Grad-CAM
# ==============================================================================

def generate_gradcam(
    model: nn.Module,
    image_path: Path,
    class_names: list[str]
):

    """
    Generate Grad-CAM for a single image.

    The same trained Epoch-8 checkpoint and the same
    inference preprocessing are used for prediction
    and Grad-CAM generation.

    Returns
    -------
    tuple
        (
            predicted_class,
            confidence,
            predicted_index,
            heatmap
        )
    """

    # ==========================================================================
    # 1. Verify Image
    # ==========================================================================

    if not image_path.exists():

        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )


    # ==========================================================================
    # 2. Load + Preprocess Image
    # ==========================================================================

    (
        input_tensor,
        rgb_image
    ) = preprocess_image(
        image_path
    )


    # ==========================================================================
    # 3. Move Input to Device
    # ==========================================================================

    input_tensor = input_tensor.to(
        DEVICE
    )


    # ==========================================================================
    # 4. Get Grad-CAM Target Layer
    # ==========================================================================

    target_layer = get_target_layer(
        model
    )


    # ==========================================================================
    # 5. Activation and Gradient Storage
    # ==========================================================================

    activations = []

    gradients = []


    # ==========================================================================
    # 6. Forward Hook
    # ==========================================================================

    def forward_hook(
        module,
        module_input,
        module_output
    ):

        activations.append(
            module_output.detach()
        )


    # ==========================================================================
    # 7. Backward Hook
    # ==========================================================================

    def backward_hook(
        module,
        grad_input,
        grad_output
    ):

        if grad_output is not None:

            gradients.append(
                grad_output[0].detach()
            )


    # ==========================================================================
    # 8. Register Hooks
    # ==========================================================================

    forward_handle = (
        target_layer.register_forward_hook(
            forward_hook
        )
    )


    backward_handle = (
        target_layer.register_full_backward_hook(
            backward_hook
        )
    )


    try:

        # ======================================================================
        # 9. Model Evaluation Mode
        # ======================================================================

        model.eval()

        model.zero_grad(
            set_to_none=True
        )


        # ======================================================================
        # 10. Forward Pass
        # ======================================================================

        output = model(
            input_tensor
        )


        # ======================================================================
        # 11. Prediction Probability
        # ======================================================================

        probabilities = torch.softmax(
            output,
            dim=1
        )


        (
            confidence_tensor,
            predicted_index_tensor
        ) = torch.max(
            probabilities,
            dim=1
        )


        predicted_index = int(
            predicted_index_tensor.item()
        )


        confidence = float(
            confidence_tensor.item()
        )


        # ======================================================================
        # 12. Select Predicted Class Score
        # ======================================================================

        target_score = output[
            0,
            predicted_index
        ]


        # ======================================================================
        # 13. Backward Pass
        # ======================================================================

        target_score.backward()


        # ======================================================================
        # 14. Verify Activation and Gradient
        # ======================================================================

        if not activations:

            raise RuntimeError(
                "Grad-CAM activation was not captured."
            )


        if not gradients:

            raise RuntimeError(
                "Grad-CAM gradient was not captured."
            )


        # ======================================================================
        # 15. Get Activation and Gradient
        # ======================================================================

        activation = activations[0]

        gradient = gradients[0]


        # ======================================================================
        # 16. Global Average Pooling
        # ======================================================================

        weights = gradient.mean(
            dim=(2, 3),
            keepdim=True
        )


        # ======================================================================
        # 17. Weighted Activation Map
        # ======================================================================

        cam = (
            weights
            *
            activation
        ).sum(
            dim=1
        )


        # ======================================================================
        # 18. ReLU
        # ======================================================================

        cam = torch.relu(
            cam
        )


        # ======================================================================
        # 19. Convert CAM to NumPy
        # ======================================================================

        heatmap = (
            cam[0]
            .detach()
            .cpu()
            .numpy()
        )


        # ======================================================================
        # 20. Normalize Heatmap
        # ======================================================================

        heatmap_min = float(
            heatmap.min()
        )


        heatmap_max = float(
            heatmap.max()
        )


        if (
            heatmap_max
            -
            heatmap_min
        ) > 1e-8:

            heatmap = (
                heatmap
                -
                heatmap_min
            ) / (
                heatmap_max
                -
                heatmap_min
            )

        else:

            heatmap = np.zeros_like(
                heatmap,
                dtype=np.float32
            )


        # ======================================================================
        # 21. Resize Heatmap to Original Image Size
        # ======================================================================

        heatmap = cv2.resize(
            heatmap,
            (
                rgb_image.shape[1],
                rgb_image.shape[0]
            ),
            interpolation=cv2.INTER_LINEAR
        )


        # ======================================================================
        # 22. Ensure Valid Range
        # ======================================================================

        heatmap = np.clip(
            heatmap,
            0.0,
            1.0
        ).astype(
            np.float32
        )


        # ======================================================================
        # 23. Get Predicted Class
        # ======================================================================

        predicted_class = (
            class_names[
                predicted_index
            ]
        )


        # ======================================================================
        # 24. Return Results
        # ======================================================================

        return (
            predicted_class,
            confidence,
            predicted_index,
            heatmap
        )


    finally:

        # ======================================================================
        # 25. Remove Hooks
        # ======================================================================

        forward_handle.remove()

        backward_handle.remove()


# ==============================================================================
# End of Phase 5 - Part 3
# ==============================================================================
# ==============================================================================
#
# PlantXAI-Net
#
# Phase 5
#
# XAI / Explainable AI
#
# Part 4
#
# Full-Test-Set Grad-CAM Generation
#
# ==============================================================================


# ==============================================================================
# XAI Coverage Configuration
# ==============================================================================

XAI_COVERAGE_MODE = "full_test_set"


# ==============================================================================
# Get All Test Images
# ==============================================================================

def get_test_images(
    test_dir: Path
) -> list[Path]:

    """
    Collect all valid test images using the same class-folder
    structure used by the test dataset.

    Returns
    -------
    list[Path]
        Sorted list of all valid test image paths.
    """

    if not test_dir.exists():

        raise FileNotFoundError(
            f"Test directory not found: {test_dir}"
        )


    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp"
    }


    class_dirs = sorted(

        [
            folder
            for folder in test_dir.iterdir()
            if folder.is_dir()
        ]

    )


    if len(class_dirs) != NUM_CLASSES:

        raise RuntimeError(

            f"Expected {NUM_CLASSES} class directories, "
            f"but found {len(class_dirs)}."

        )


    all_images = []


    for class_dir in class_dirs:

        class_images = sorted(

            [
                image_path
                for image_path in class_dir.iterdir()
                if (
                    image_path.is_file()
                    and
                    image_path.suffix.lower()
                    in image_extensions
                )
            ]

        )


        if not class_images:

            print(

                f"WARNING: No valid images found in "
                f"{class_dir.name}"

            )


        all_images.extend(
            class_images
        )


    if not all_images:

        raise RuntimeError(
            "No valid test images were found."
        )


    return all_images


# ==============================================================================
# Generate Grad-CAM For Complete Test Set
# ==============================================================================

def generate_batch_gradcam(
    model: nn.Module,
    test_dir: Path,
    class_names: list[str]
) -> None:

    """
    Generate Grad-CAM for every image in the complete test set.

    The same loaded Epoch-8 checkpoint and the same class mapping
    are used throughout the evaluation.

    Grad-CAM heatmaps are calculated in memory and are not saved.
    Numerical XAI results are saved to CSV.
    """

    # --------------------------------------------------------------------------
    # Get Complete Test Set
    # --------------------------------------------------------------------------

    image_paths = get_test_images(
        test_dir
    )


    expected_images = len(
        image_paths
    )


    # --------------------------------------------------------------------------
    # Verify Class Mapping
    # --------------------------------------------------------------------------

    test_dataset = datasets.ImageFolder(
        root=test_dir
    )


    if test_dataset.classes != class_names:

        raise RuntimeError(

            "Test-set class mapping does not match "
            "the training class mapping."

        )


    # --------------------------------------------------------------------------
    # Header
    # --------------------------------------------------------------------------

    print()

    print("=" * 80)

    print(
        "FULL-TEST-SET GRAD-CAM ANALYSIS"
    )

    print("=" * 80)

    print()


    print(
        f"Classes Represented : "
        f"{len(class_names)}"
    )


    print(
        f"Images Evaluated    : "
        f"{expected_images}"
    )


    print(
        f"Coverage Mode       : "
        f"{XAI_COVERAGE_MODE}"
    )


    print(
        "Sampling Strategy   : "
        "None - Complete Test Set"
    )


    print()


    # --------------------------------------------------------------------------
    # Counters
    # --------------------------------------------------------------------------

    successful = 0

    failed = 0

    correct_predictions = 0

    incorrect_predictions = 0

    results = []


    # --------------------------------------------------------------------------
    # Process All Images
    # --------------------------------------------------------------------------

    for index, image_path in enumerate(
        image_paths,
        start=1
    ):

        print(

            f"[{index}/{expected_images}] "
            f"Processing: {image_path.name}"

        )


        try:

            # ------------------------------------------------------------------
            # Grad-CAM Prediction
            # ------------------------------------------------------------------

            (
                predicted_class,
                confidence,
                predicted_index,
                heatmap

            ) = generate_gradcam(

                model,
                image_path,
                class_names

            )


            # ------------------------------------------------------------------
            # Resolve Actual Class Using ImageFolder Mapping
            # ------------------------------------------------------------------
            #
            # This avoids relying only on folder-name text and guarantees
            # consistency with the 38-class ImageFolder mapping.
            # ------------------------------------------------------------------

            actual_class = image_path.parent.name


            if actual_class not in class_names:

                raise RuntimeError(

                    f"Unknown test class folder: "
                    f"{actual_class}"

                )


            # ------------------------------------------------------------------
            # Prediction Correctness
            # ------------------------------------------------------------------

            correct = (

                actual_class
                ==
                predicted_class

            )


            if correct:

                correct_predictions += 1

            else:

                incorrect_predictions += 1


            # ------------------------------------------------------------------
            # Save Numerical Result
            # ------------------------------------------------------------------

            results.append(

                {

                    "image":
                    image_path.name,

                    "image_path":
                    str(image_path),

                    "actual_class":
                    actual_class,

                    "predicted_class":
                    predicted_class,

                    "predicted_index":
                    predicted_index,

                    "confidence":
                    confidence,

                    "correct":
                    correct,

                    "xai_coverage":
                    "full_test_set"

                }

            )


            successful += 1


            # ------------------------------------------------------------------
            # Console Output
            # ------------------------------------------------------------------

            print(

                f"    Actual     : "
                f"{actual_class}"

            )


            print(

                f"    Prediction : "
                f"{predicted_class}"

            )


            print(

                f"    Confidence : "
                f"{confidence * 100:.2f}%"

            )


            print(

                f"    Correct    : "
                f"{correct}"

            )


            print(
                "    Status     : SUCCESS"
            )


            print()


        except Exception as error:

            failed += 1


            print(
                "    Status     : FAILED"
            )


            print(

                f"    Error      : "
                f"{error}"

            )


            print()


    # ==========================================================================
    # Save Full-Test XAI Results
    # ==========================================================================

    if not results:

        raise RuntimeError(

            "No XAI results were successfully generated."

        )


    results_df = pd.DataFrame(
        results
    )


    total_processed = len(
        results_df
    )


    # --------------------------------------------------------------------------
    # Verify Complete Coverage
    # --------------------------------------------------------------------------

    if total_processed != expected_images:

        raise RuntimeError(

            f"XAI processing incomplete. "
            f"Expected {expected_images} results, "
            f"but generated {total_processed}."

        )


    # --------------------------------------------------------------------------
    # Verify Image Count
    # --------------------------------------------------------------------------

    if successful != expected_images:

        raise RuntimeError(

            f"Successful XAI processing count mismatch. "
            f"Expected {expected_images}, "
            f"but processed {successful} successfully."

        )


    # --------------------------------------------------------------------------
    # Calculate Full-Test Prediction Accuracy
    # --------------------------------------------------------------------------

    full_test_accuracy = (

        correct_predictions
        /
        total_processed

    )


    # --------------------------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------------------------

    summary_path = (

        XAI_RESULTS_DIR
        /
        "xai_full_test_results.csv"

    )


    results_df.to_csv(

        summary_path,

        index=False

    )


    # ==========================================================================
    # Final Output
    # ==========================================================================

    print("=" * 80)

    print(
        "FULL-TEST-SET GRAD-CAM COMPLETED"
    )

    print("=" * 80)

    print()


    print(
        f"Classes Represented  : "
        f"{len(class_names)}"
    )


    print(
        f"Images Evaluated     : "
        f"{expected_images}"
    )


    print(
        f"Successful           : "
        f"{successful}"
    )


    print(
        f"Failed               : "
        f"{failed}"
    )


    print()


    print(
        f"Correct Predictions  : "
        f"{correct_predictions}"
    )


    print(
        f"Incorrect Predictions: "
        f"{incorrect_predictions}"
    )


    print(
        f"Prediction Accuracy  : "
        f"{full_test_accuracy * 100:.2f}%"
    )


    print()


    print(
        f"Results Saved        : "
        f"{summary_path}"
    )


    print()

    print("=" * 80)

    print(
        "Phase 5 - Part 4 Completed Successfully"
    )

    print("=" * 80)

    print()


# ==============================================================================
# End of Phase 5 - Part 4
# ==============================================================================
# ==============================================================================
# PlantXAI-Net
# Phase 5
# XAI / Explainable AI
# Part 5
# Full-Test XAI Results Analysis
# ==============================================================================


# ==============================================================================
# XAI Analysis Configuration
# ==============================================================================

XAI_BATCH_RESULTS_PATH = (
    XAI_RESULTS_DIR
    /
    "xai_full_test_results.csv"
)


XAI_ANALYSIS_PATH = (
    XAI_RESULTS_DIR
    /
    "xai_full_test_analysis.csv"
)


# ==============================================================================
# Analyze Full-Test XAI Results
# ==============================================================================

def analyze_xai_results() -> None:

    """
    Analyze Grad-CAM results generated for the complete test set.

    Grad-CAM is calculated during Part 4, but visualization images
    are not saved to disk.
    """

    # --------------------------------------------------------------------------
    # Verify Results File
    # --------------------------------------------------------------------------

    if not XAI_BATCH_RESULTS_PATH.exists():

        raise FileNotFoundError(
            f"Full-test XAI results not found: "
            f"{XAI_BATCH_RESULTS_PATH}"
        )


    # --------------------------------------------------------------------------
    # Load Results
    # --------------------------------------------------------------------------

    results_df = pd.read_csv(
        XAI_BATCH_RESULTS_PATH
    )


    if results_df.empty:

        raise RuntimeError(
            "Full-test XAI results file is empty."
        )


    # --------------------------------------------------------------------------
    # Required Columns Check
    # --------------------------------------------------------------------------

    required_columns = [
        "image",
        "image_path",
        "actual_class",
        "predicted_class",
        "confidence",
        "correct",
        "xai_coverage"
    ]


    missing_columns = [
        column
        for column in required_columns
        if column not in results_df.columns
    ]


    if missing_columns:

        raise RuntimeError(
            "Missing required columns in XAI results: "
            +
            ", ".join(missing_columns)
        )


    # --------------------------------------------------------------------------
    # Recalculate Prediction Correctness
    # --------------------------------------------------------------------------

    results_df["correct"] = (
        results_df["actual_class"]
        ==
        results_df["predicted_class"]
    )


    # --------------------------------------------------------------------------
    # Confidence Percentage
    # --------------------------------------------------------------------------

    results_df["confidence_percent"] = (
        results_df["confidence"]
        *
        100.0
    )


    # --------------------------------------------------------------------------
    # Summary Statistics
    # --------------------------------------------------------------------------

    total_images = len(
        results_df
    )


    correct_predictions = int(
        results_df["correct"].sum()
    )


    incorrect_predictions = (
        total_images
        -
        correct_predictions
    )


    prediction_accuracy = (
        correct_predictions
        /
        total_images
    )


    average_confidence = (
        results_df["confidence_percent"]
        .mean()
    )


    minimum_confidence = (
        results_df["confidence_percent"]
        .min()
    )


    maximum_confidence = (
        results_df["confidence_percent"]
        .max()
    )


    # --------------------------------------------------------------------------
    # Confidence Categories
    # --------------------------------------------------------------------------

    def confidence_category(
        confidence: float
    ) -> str:

        if confidence >= 90:

            return "High"

        elif confidence >= 70:

            return "Medium"

        else:

            return "Low"


    results_df["confidence_category"] = (
        results_df["confidence_percent"]
        .apply(
            confidence_category
        )
    )


    # --------------------------------------------------------------------------
    # Confidence Distribution
    # --------------------------------------------------------------------------

    high_confidence = int(
        (
            results_df["confidence_category"]
            ==
            "High"
        ).sum()
    )


    medium_confidence = int(
        (
            results_df["confidence_category"]
            ==
            "Medium"
        ).sum()
    )


    low_confidence = int(
        (
            results_df["confidence_category"]
            ==
            "Low"
        ).sum()
    )


    # --------------------------------------------------------------------------
    # Save Detailed Analysis
    # --------------------------------------------------------------------------

    results_df.to_csv(
        XAI_ANALYSIS_PATH,
        index=False
    )


    # ==============================================================================
    # Console Output
    # ==============================================================================

    print()

    print("=" * 80)

    print(
        "FULL-TEST XAI RESULTS ANALYSIS"
    )

    print("=" * 80)

    print()


    print(
        f"Images Evaluated      : "
        f"{total_images}"
    )


    print(
        f"Correct Predictions   : "
        f"{correct_predictions}"
    )


    print(
        f"Incorrect Predictions : "
        f"{incorrect_predictions}"
    )


    print(
        f"Prediction Accuracy   : "
        f"{prediction_accuracy * 100:.2f}%"
    )


    print()


    print(
        f"Average Confidence    : "
        f"{average_confidence:.2f}%"
    )


    print(
        f"Lowest Confidence     : "
        f"{minimum_confidence:.2f}%"
    )


    print(
        f"Highest Confidence    : "
        f"{maximum_confidence:.2f}%"
    )


    print()


    # --------------------------------------------------------------------------
    # Confidence Distribution
    # --------------------------------------------------------------------------

    print(
        "Confidence Distribution"
    )

    print(
        "-" * 80
    )


    print(
        f"High   (>= 90%) : "
        f"{high_confidence}"
    )


    print(
        f"Medium (70-90%) : "
        f"{medium_confidence}"
    )


    print(
        f"Low    (< 70%)  : "
        f"{low_confidence}"
    )


    print()


    # --------------------------------------------------------------------------
    # Low Confidence Cases
    # --------------------------------------------------------------------------

    low_confidence_df = results_df[
        results_df["confidence_percent"]
        <
        70
    ]


    print(
        f"Low Confidence Cases : "
        f"{len(low_confidence_df)}"
    )


    if not low_confidence_df.empty:

        print(
            "-" * 80
        )


        for _, row in low_confidence_df.iterrows():

            print(
                f"Image      : "
                f"{row['image']}"
            )


            print(
                f"Actual     : "
                f"{row['actual_class']}"
            )


            print(
                f"Predicted  : "
                f"{row['predicted_class']}"
            )


            print(
                f"Confidence : "
                f"{row['confidence_percent']:.2f}%"
            )


            print()


    # --------------------------------------------------------------------------
    # Save Confirmation
    # --------------------------------------------------------------------------

    print(
        f"Analysis Saved       : "
        f"{XAI_ANALYSIS_PATH}"
    )


    print()

    print("=" * 80)

    print(
        "Phase 5 - Part 5 Completed Successfully"
    )

    print("=" * 80)

    print()


# ==============================================================================
# End of Phase 5 - Part 5
# ==============================================================================


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":

    verify_gradcam_setup()

    class_names = load_class_names()

    model = load_trained_model()

    generate_batch_gradcam(
        model,
        TEST_DIR,
        class_names
    )

    analyze_xai_results()


# ==============================================================================
# End of Phase 5
# ==============================================================================