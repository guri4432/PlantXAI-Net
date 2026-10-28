# ==============================================================================
#
# PlantXAI-Net
#
# Phase 6
#
# Severity Estimation
#
# Part 1
#
# Severity Configuration
#
# ==============================================================================


# ==============================================================================
# Imports
# ==============================================================================


from pathlib import Path

import pandas as pd
import cv2
import numpy as np

from dataclasses import dataclass


# ==============================================================================
# Project Configuration
# ==============================================================================

PROJECT_ROOT = (

    Path(__file__).resolve().parent.parent

)


# ==============================================================================
# Model Configuration
# ==============================================================================

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


TEST_DIR = (

    DATASET_ROOT

    / "split"

    / "test"

)


# ==============================================================================
# XAI Configuration
# ==============================================================================

XAI_RESULTS_DIR = (

    PROJECT_ROOT

    / "results"

    / "xai"

)


# ==============================================================================
# Severity Results Configuration
# ==============================================================================

SEVERITY_RESULTS_DIR = (

    PROJECT_ROOT

    / "results"

    / "severity"

)


SEVERITY_RESULTS_DIR.mkdir(

    parents=True,

    exist_ok=True

)


# ==============================================================================
# Calibration Configuration
# ==============================================================================

CALIBRATION_DIR = (

    SEVERITY_RESULTS_DIR

    / "calibration"

)


CALIBRATION_DIR.mkdir(

    parents=True,

    exist_ok=True

)


CALIBRATION_FILE = (

    CALIBRATION_DIR

    / "severity_calibration.csv"

)

# ==============================================================================
# Severity Levels
# ==============================================================================

SEVERITY_MILD = "Mild"

SEVERITY_MODERATE = "Moderate"

SEVERITY_SEVERE = "Severe"


SEVERITY_LEVELS = [

    SEVERITY_MILD,

    SEVERITY_MODERATE,

    SEVERITY_SEVERE

]

# ==============================================================================
# Calibration Severity Levels
# ==============================================================================

CALIBRATION_LEVELS = [

    SEVERITY_MILD,

    SEVERITY_MODERATE,

    SEVERITY_SEVERE

]
# ==============================================================================
# Severity Threshold Configuration
# ==============================================================================

# These thresholds are intentionally not fixed yet.
#
# They will be calibrated using reference severity data in a later part.
#
# Therefore, do NOT treat these values as scientifically validated
# severity thresholds.


MILD_THRESHOLD = None

MODERATE_THRESHOLD = None

SEVERE_THRESHOLD = None


# ==============================================================================
# Severity Result Structure
# ==============================================================================

@dataclass
class SeverityResult:
    """
    Store severity estimation results.
    """

    image_name: str

    predicted_disease: str

    confidence: float

    affected_area_percent: float | None

    severity_score: float | None

    severity_level: str | None


# ==============================================================================
# Severity Result Validation
# ==============================================================================

def validate_severity_result(

    result: SeverityResult

) -> bool:

    """
    Validate a severity result.

    Parameters
    ----------
    result : SeverityResult
        Severity estimation result.

    Returns
    -------
    bool
        True when the result structure is valid.
    """

    if not result.image_name:

        return False


    if not result.predicted_disease:

        return False


    if not (

        0.0

        <=

        result.confidence

        <=

        1.0

    ):

        return False


    if result.affected_area_percent is not None:

        if not (

            0.0

            <=

            result.affected_area_percent

            <=

            100.0

        ):

            return False


    if result.severity_score is not None:

        if not (

            0.0

            <=

            result.severity_score

            <=

            100.0

        ):

            return False


    if result.severity_level is not None:

        if result.severity_level not in SEVERITY_LEVELS:

            return False


    return True


# ==============================================================================
# Configuration Verification
# ==============================================================================

def verify_severity_configuration() -> None:
    """
    Verify Phase 6 severity configuration.
    """

    print()

    print("=" * 80)

    print("PlantXAI-Net")

    print("Phase 6 - Severity Estimation")

    print("Part 1 - Severity Configuration Verification")

    print("=" * 80)

    print()

    print(

        f"Project Root          : {PROJECT_ROOT}"

    )

    print(

        f"Dataset Root          : {DATASET_ROOT}"

    )

    print(

        f"Test Directory        : {TEST_DIR}"

    )

    print(

        f"XAI Results Directory : {XAI_RESULTS_DIR}"

    )

    print(

        f"Severity Results      : {SEVERITY_RESULTS_DIR}"

    )

    print()

    print(

        f"Severity Levels       : {', '.join(SEVERITY_LEVELS)}"

    )

    print()

    print(

        "Severity Thresholds   : Pending Calibration"

    )

    print()

    print("=" * 80)

    print(

        "Phase 6 - Part 1 Configuration Ready"

    )

    print("=" * 80)

    print()


# ==============================================================================
# End of Phase 6 - Part 1
# ==============================================================================

# ==============================================================================
#
# ==============================================================================
#
# PlantXAI-Net
#
# Phase 6
# Severity Estimation
#
# Part 2
#
# Robust Leaf Region + Affected Region Measurement
#
# ==============================================================================


# ==============================================================================
# Additional Imports
# ==============================================================================

import numpy as np

from collections import deque

from PIL import Image
from PIL import ImageFilter


# ==============================================================================
# Image Configuration
# ==============================================================================

HEATMAP_THRESHOLD = 0.50

MIN_LEAF_COMPONENT_SIZE = 500


# ==============================================================================
# Load RGB Image
# ==============================================================================

def load_rgb_image(
    image_path: Path
) -> np.ndarray:
    """
    Load an image as an RGB NumPy array.
    """

    if not image_path.exists():

        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    image = Image.open(
        image_path
    ).convert(
        "RGB"
    )

    return np.asarray(
        image,
        dtype=np.uint8
    )


# ==============================================================================
# Largest Connected Component
# ==============================================================================

def get_largest_component(
    mask: np.ndarray
) -> np.ndarray:
    """
    Extract the largest connected component from a binary mask.

    This helps remove isolated background regions.
    """

    height, width = mask.shape

    visited = np.zeros(
        mask.shape,
        dtype=bool
    )

    largest_component = np.zeros(
        mask.shape,
        dtype=bool
    )

    largest_size = 0

    for y in range(height):

        for x in range(width):

            if not mask[y, x]:
                continue

            if visited[y, x]:
                continue

            queue = deque()

            queue.append(
                (y, x)
            )

            visited[y, x] = True

            component = []

            while queue:

                current_y, current_x = queue.popleft()

                component.append(
                    (current_y, current_x)
                )

                neighbors = (
                    (current_y - 1, current_x),
                    (current_y + 1, current_x),
                    (current_y, current_x - 1),
                    (current_y, current_x + 1)
                )

                for next_y, next_x in neighbors:

                    if next_y < 0:
                        continue

                    if next_y >= height:
                        continue

                    if next_x < 0:
                        continue

                    if next_x >= width:
                        continue

                    if visited[next_y, next_x]:
                        continue

                    if not mask[next_y, next_x]:
                        continue

                    visited[next_y, next_x] = True

                    queue.append(
                        (next_y, next_x)
                    )

            component_size = len(
                component
            )

            if component_size > largest_size:

                largest_size = component_size

                largest_component.fill(False)

                for component_y, component_x in component:

                    largest_component[
                        component_y,
                        component_x
                    ] = True

    return largest_component


# ==============================================================================
# Create Robust Leaf Mask
# ==============================================================================

def create_leaf_mask(
    image: np.ndarray
) -> np.ndarray:
    """
    Estimate the visible leaf region.

    Primary segmentation:
        Green vegetation + saturation characteristics.

    Fallback segmentation:
        A broader foreground/color mask is used when the
        primary segmentation produces an extremely small
        leaf region.

    This remains an image-based leaf-region proxy and is
    not considered ground-truth segmentation.
    """

    # ==========================================================================
    # Convert image to HSV
    # ==========================================================================

    image_pil = Image.fromarray(
        image
    )

    hsv_image = image_pil.convert(
        "HSV"
    )

    hsv = np.asarray(
        hsv_image,
        dtype=np.uint8
    )

    saturation = hsv[:, :, 1]

    value = hsv[:, :, 2]

    red = image[:, :, 0].astype(
        np.float32
    )

    green = image[:, :, 1].astype(
        np.float32
    )

    blue = image[:, :, 2].astype(
        np.float32
    )


    # ==========================================================================
    # PRIMARY SEGMENTATION
    # ==========================================================================

    # --------------------------------------------------------------------------
    # Green / vegetation mask
    # --------------------------------------------------------------------------

    green_mask = (
        (green > red * 1.02)
        &
        (green > blue * 1.02)
        &
        (green > 35)
    )


    # --------------------------------------------------------------------------
    # Saturated leaf-color mask
    # --------------------------------------------------------------------------

    saturated_mask = (
        (saturation > 35)
        &
        (value > 30)
        &
        (value < 250)
    )


    # --------------------------------------------------------------------------
    # Combine vegetation characteristics
    # --------------------------------------------------------------------------

    primary_mask = (
        green_mask
        &
        saturated_mask
    )


    # --------------------------------------------------------------------------
    # Morphological closing
    # --------------------------------------------------------------------------

    mask_image = Image.fromarray(
        (
            primary_mask.astype(
                np.uint8
            )
            *
            255
        )
    )


    mask_image = mask_image.filter(
        ImageFilter.MaxFilter(
            size=9
        )
    )


    mask_image = mask_image.filter(
        ImageFilter.MinFilter(
            size=9
        )
    )


    primary_mask = (
        np.asarray(
            mask_image
        )
        >
        127
    )


    # --------------------------------------------------------------------------
    # Largest connected primary leaf component
    # --------------------------------------------------------------------------

    primary_leaf_mask = get_largest_component(
        primary_mask
    )


    primary_pixels = int(
        primary_leaf_mask.sum()
    )


    # ==========================================================================
    # PRIMARY SEGMENTATION SUCCESS
    # ==========================================================================

    if primary_pixels >= MIN_LEAF_COMPONENT_SIZE:

        return primary_leaf_mask


    # ==========================================================================
    # FALLBACK SEGMENTATION
    # ==========================================================================

    """
    The primary segmentation is intentionally conservative.

    Diseased leaves may become:
        - yellow
        - brown
        - pale
        - weakly green
        - partially discolored

    Such pixels may not satisfy the strict green-dominance
    condition. Therefore, a broader color/foreground mask
    is attempted only when the primary segmentation fails.
    """


    # --------------------------------------------------------------------------
    # Broad foreground/color mask
    # --------------------------------------------------------------------------

    fallback_mask = (
        (saturation > 20)
        &
        (value > 25)
        &
        (value < 250)
    )


    # --------------------------------------------------------------------------
    # Preserve chromatic green information
    # --------------------------------------------------------------------------

    fallback_green_mask = (
        (green > red * 1.01)
        &
        (green > blue * 1.01)
        &
        (green > 30)
    )


    # --------------------------------------------------------------------------
    # Combine broader foreground characteristics
    # --------------------------------------------------------------------------

    fallback_mask = (
        fallback_mask
        |
        fallback_green_mask
    )


    # --------------------------------------------------------------------------
    # Morphological processing
    # --------------------------------------------------------------------------

    fallback_image = Image.fromarray(
        (
            fallback_mask.astype(
                np.uint8
            )
            *
            255
        )
    )


    # Closing helps reconnect fragmented leaf regions.

    fallback_image = fallback_image.filter(
        ImageFilter.MaxFilter(
            size=11
        )
    )


    fallback_image = fallback_image.filter(
        ImageFilter.MinFilter(
            size=7
        )
    )


    fallback_mask = (
        np.asarray(
            fallback_image
        )
        >
        127
    )


    # --------------------------------------------------------------------------
    # Largest connected fallback component
    # --------------------------------------------------------------------------

    fallback_leaf_mask = get_largest_component(
        fallback_mask
    )


    fallback_pixels = int(
        fallback_leaf_mask.sum()
    )


    # ==========================================================================
    # FALLBACK SUCCESS
    # ==========================================================================

    if fallback_pixels >= MIN_LEAF_COMPONENT_SIZE:

        return fallback_leaf_mask


    # ==========================================================================
    # SECOND FALLBACK
    # ==========================================================================

    """
    If both color-based masks remain too small, use a broader
    non-background chromatic mask.

    This stage is intentionally used only for the difficult
    cases that failed the previous segmentation stages.
    """

    second_fallback_mask = (
        (
            saturation > 12
        )
        &
        (
            value > 20
        )
        &
        (
            value < 252
        )
    )


    # --------------------------------------------------------------------------
    # Remove extremely weak neutral pixels
    # --------------------------------------------------------------------------

    chroma_strength = np.maximum.reduce(
        [
            np.abs(green - red),
            np.abs(green - blue),
            np.abs(red - blue)
        ]
    )


    second_fallback_mask = (
        second_fallback_mask
        &
        (
            chroma_strength > 5
        )
    )


    # --------------------------------------------------------------------------
    # Morphological processing
    # --------------------------------------------------------------------------

    second_fallback_image = Image.fromarray(
        (
            second_fallback_mask.astype(
                np.uint8
            )
            *
            255
        )
    )


    second_fallback_image = second_fallback_image.filter(
        ImageFilter.MaxFilter(
            size=11
        )
    )


    second_fallback_image = second_fallback_image.filter(
        ImageFilter.MinFilter(
            size=7
        )
    )


    second_fallback_mask = (
        np.asarray(
            second_fallback_image
        )
        >
        127
    )


    # --------------------------------------------------------------------------
    # Largest connected component
    # --------------------------------------------------------------------------

    second_leaf_mask = get_largest_component(
        second_fallback_mask
    )


    second_pixels = int(
        second_leaf_mask.sum()
    )


    # ==========================================================================
    # SECOND FALLBACK SUCCESS
    # ==========================================================================

    if second_pixels >= MIN_LEAF_COMPONENT_SIZE:

        return second_leaf_mask


    # ==========================================================================
    # FINAL FAILURE
    # ==========================================================================

    raise RuntimeError(
        "Leaf segmentation failed after primary and "
        "fallback segmentation attempts."
    )


# ==============================================================================
# Load Grad-CAM Heatmap
# ==============================================================================

def load_gradcam_heatmap(
    image_path: Path
) -> np.ndarray:
    """
    Load the grayscale Grad-CAM heatmap generated by Phase 5.
    """

    heatmap_path = (
        XAI_RESULTS_DIR
        /
        f"{image_path.stem}_heatmap.png"
    )


    if not heatmap_path.exists():

        raise FileNotFoundError(
            f"Grad-CAM heatmap not found: {heatmap_path}\n"
            "Run Phase 5 Grad-CAM generation first."
        )


    heatmap_image = Image.open(
        heatmap_path
    ).convert(
        "L"
    )


    heatmap = (
        np.asarray(
            heatmap_image,
            dtype=np.float32
        )
        /
        255.0
    )


    return heatmap


# ==============================================================================
# Calculate Affected Region
# ==============================================================================

def calculate_affected_region(
    leaf_mask: np.ndarray,
    heatmap: np.ndarray,
    threshold: float = HEATMAP_THRESHOLD
) -> tuple[int, int, float]:
    """
    Calculate Grad-CAM highlighted area inside the estimated
    leaf region.

    The result is an affected-area proxy and not ground-truth
    disease severity.
    """

    if leaf_mask.shape != heatmap.shape:

        raise ValueError(
            "Leaf mask and Grad-CAM heatmap dimensions do not match."
        )


    # ==========================================================================
    # Grad-CAM activation mask
    # ==========================================================================

    activation_mask = (
        heatmap >= threshold
    )


    # ==========================================================================
    # Restrict activation strictly to leaf region
    # ==========================================================================

    affected_mask = (
        activation_mask
        &
        leaf_mask
    )


    # ==========================================================================
    # Pixel Counts
    # ==========================================================================

    leaf_pixels = int(
        leaf_mask.sum()
    )


    affected_pixels = int(
        affected_mask.sum()
    )


    if leaf_pixels == 0:

        affected_area_percent = 0.0

    else:

        affected_area_percent = (
            affected_pixels
            /
            leaf_pixels
        ) * 100.0


    return (
        leaf_pixels,
        affected_pixels,
        affected_area_percent
    )


# ==============================================================================
# Save Improved Region Visualization
# ==============================================================================

def save_region_visualization(
    image: np.ndarray,
    leaf_mask: np.ndarray,
    heatmap: np.ndarray,
    image_path: Path
) -> Path:
    """
    Save visualization showing only Grad-CAM activation
    inside the estimated leaf region.
    """

    visualization = image.copy()


    # ==========================================================================
    # Grad-CAM activation mask
    # ==========================================================================

    activation_mask = (
        heatmap >= HEATMAP_THRESHOLD
    )


    # ==========================================================================
    # Restrict activation to leaf region
    # ==========================================================================

    affected_mask = (
        activation_mask
        &
        leaf_mask
    )


    # ==========================================================================
    # Create red overlay
    # ==========================================================================

    overlay = visualization.copy()


    overlay[affected_mask] = np.array(
        [255, 0, 0],
        dtype=np.uint8
    )


    # ==========================================================================
    # Blend only affected leaf pixels
    # ==========================================================================

    visualization[affected_mask] = (
        0.45
        *
        visualization[affected_mask].astype(
            np.float32
        )
        +
        0.55
        *
        overlay[affected_mask].astype(
            np.float32
        )
    ).astype(
        np.uint8
    )


    # ==========================================================================
    # Explicitly preserve background
    # ==========================================================================

    background_mask = ~leaf_mask


    visualization[background_mask] = (
        image[background_mask]
    )


    # ==========================================================================
    # Save visualization
    # ==========================================================================

    output_path = (
        SEVERITY_RESULTS_DIR
        /
        f"{image_path.stem}_affected_region.png"
    )


    Image.fromarray(
        visualization
    ).save(
        output_path
    )


    return output_path


# ==============================================================================
# Analyze Single Image
# ==============================================================================

def analyze_image_regions(
    image_path: Path
) -> tuple[int, int, float, Path]:
    """
    Analyze leaf and Grad-CAM affected regions.
    """

    # --------------------------------------------------------------------------
    # Load image
    # --------------------------------------------------------------------------

    image = load_rgb_image(
        image_path
    )


    # --------------------------------------------------------------------------
    # Generate robust leaf mask
    # --------------------------------------------------------------------------

    leaf_mask = create_leaf_mask(
        image
    )


    # --------------------------------------------------------------------------
    # Load Grad-CAM heatmap
    # --------------------------------------------------------------------------

    heatmap = load_gradcam_heatmap(
        image_path
    )


    # --------------------------------------------------------------------------
    # Calculate affected region
    # --------------------------------------------------------------------------

    (
        leaf_pixels,
        affected_pixels,
        affected_area_percent
    ) = calculate_affected_region(
        leaf_mask,
        heatmap
    )


    # --------------------------------------------------------------------------
    # Save visualization
    # --------------------------------------------------------------------------

    visualization_path = save_region_visualization(
        image,
        leaf_mask,
        heatmap,
        image_path
    )


    return (
        leaf_pixels,
        affected_pixels,
        affected_area_percent,
        visualization_path
    )


# ==============================================================================
# End of Phase 6 - Part 2
# ==============================================================================

# ==============================================================================
# ==============================================================================
# PHASE 6 - PART 3
# FULL-TEST-SET SEVERITY CALIBRATION DATASET
# ==============================================================================
#
# IMPORTANT:
# - Uses complete Phase 5 XAI test-set results.
# - No 76-image restriction.
# - No 2-images-per-class restriction.
# - Grad-CAM is recalculated in memory.
# - No Grad-CAM heatmap PNG files are required.
# - No severity visualization PNG files are generated.
# - Reference severity fields remain blank until independent annotation.
# ==============================================================================


# ==============================================================================
# Locate Full-Test XAI Results
# ==============================================================================

def find_xai_batch_results() -> Path:

    """
    Locate the complete Phase 5 XAI results CSV.

    The complete test-set result file is preferred.
    """

    candidates = [

        XAI_RESULTS_DIR
        /
        "xai_full_test_results.csv",

        XAI_RESULTS_DIR
        /
        "xai_full_test_analysis.csv"

    ]


    for file_path in candidates:

        if file_path.exists():

            return file_path


    raise FileNotFoundError(

        "Full-test Phase 5 XAI results CSV not found.\n"

        "Expected one of:\n"

        f"  {candidates[0]}\n"

        f"  {candidates[1]}"

    )


# ==============================================================================
# Load EfficientNet-B0 For Grad-CAM
# ==============================================================================

def load_severity_gradcam_model():

    """
    Load the same EfficientNet-B0 architecture and checkpoint
    used by Phase 5.

    Imports are local so the rest of the severity script does
    not require PyTorch at module-import time.
    """

    import torch

    import torch.nn as nn

    from torchvision import models


    # --------------------------------------------------------------------------
    # Device
    # --------------------------------------------------------------------------

    device = torch.device(

        "cuda"
        if torch.cuda.is_available()
        else "cpu"

    )


    # --------------------------------------------------------------------------
    # Checkpoint
    # --------------------------------------------------------------------------

    checkpoint_path = (

        PROJECT_ROOT
        /
        "models"
        /
        "checkpoints"
        /
        "best_model.pth"

    )


    if not checkpoint_path.exists():

        raise FileNotFoundError(

            f"Best model checkpoint not found: "
            f"{checkpoint_path}"

        )


    # --------------------------------------------------------------------------
    # Build EfficientNet-B0
    # --------------------------------------------------------------------------

    model = models.efficientnet_b0(

        weights=None

    )


    # --------------------------------------------------------------------------
    # Classification Head
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


    # --------------------------------------------------------------------------
    # Load Checkpoint
    # --------------------------------------------------------------------------

    checkpoint = torch.load(

        checkpoint_path,

        map_location=device

    )


    if "model_state_dict" not in checkpoint:

        raise KeyError(

            "model_state_dict not found in checkpoint."

        )


    model.load_state_dict(

        checkpoint["model_state_dict"]

    )


    model = model.to(

        device

    )


    model.eval()


    return (

        model,
        device

    )


# ==============================================================================
# Generate Grad-CAM Heatmap In Memory
# ==============================================================================

def generate_severity_gradcam(

    model,
    device,
    image_path: Path

) -> np.ndarray:

    """
    Generate a normalized Grad-CAM heatmap for one image.

    The heatmap is returned in memory and is NOT saved as an image.
    """

    import torch


    # --------------------------------------------------------------------------
    # Load Image
    # --------------------------------------------------------------------------

    image = cv2.imread(

        str(image_path)

    )


    if image is None:

        raise ValueError(

            f"Unable to read image: {image_path}"

        )


    # --------------------------------------------------------------------------
    # BGR -> RGB
    # --------------------------------------------------------------------------

    rgb_image = cv2.cvtColor(

        image,

        cv2.COLOR_BGR2RGB

    )


    # --------------------------------------------------------------------------
    # Resize
    # --------------------------------------------------------------------------

    resized = cv2.resize(

        rgb_image,

        (
            IMAGE_SIZE,
            IMAGE_SIZE
        ),

        interpolation=cv2.INTER_AREA

    )


    # --------------------------------------------------------------------------
    # Convert To Float
    # --------------------------------------------------------------------------

    input_image = (

        resized.astype(

            np.float32

        )

        /

        255.0

    )


    # --------------------------------------------------------------------------
    # ImageNet Normalization
    # --------------------------------------------------------------------------

    mean = np.array(

        (
            0.485,
            0.456,
            0.406
        ),

        dtype=np.float32

    )


    std = np.array(

        (
            0.229,
            0.224,
            0.225
        ),

        dtype=np.float32

    )


    input_image = (

        input_image
        -
        mean

    )


    input_image = (

        input_image
        /
        std

    )


    # --------------------------------------------------------------------------
    # HWC -> CHW
    # --------------------------------------------------------------------------

    input_image = np.transpose(

        input_image,

        (
            2,
            0,
            1
        )

    )


    input_tensor = torch.from_numpy(

        input_image

    ).unsqueeze(

        0

    ).to(

        device

    )


    # --------------------------------------------------------------------------
    # Target Layer
    # --------------------------------------------------------------------------

    target_layer = model.features[-1]


    activations = []

    gradients = []


    # --------------------------------------------------------------------------
    # Forward Hook
    # --------------------------------------------------------------------------

    def forward_hook(

        module,
        module_input,
        module_output

    ):

        activations.append(

            module_output.detach()

        )


    # --------------------------------------------------------------------------
    # Backward Hook
    # --------------------------------------------------------------------------

    def backward_hook(

        module,
        grad_input,
        grad_output

    ):

        if grad_output:

            gradients.append(

                grad_output[0].detach()

            )


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

        # ----------------------------------------------------------------------
        # Clear Gradients
        # ----------------------------------------------------------------------

        model.zero_grad(

            set_to_none=True

        )


        # ----------------------------------------------------------------------
        # Forward Pass
        # ----------------------------------------------------------------------

        output = model(

            input_tensor

        )


        # ----------------------------------------------------------------------
        # Predicted Class
        # ----------------------------------------------------------------------

        predicted_index = int(

            torch.argmax(

                output,
                dim=1

            ).item()

        )


        # ----------------------------------------------------------------------
        # Backward Pass
        # ----------------------------------------------------------------------

        target_score = output[

            0,
            predicted_index

        ]


        target_score.backward()


        # ----------------------------------------------------------------------
        # Validate Hooks
        # ----------------------------------------------------------------------

        if not activations:

            raise RuntimeError(

                "Grad-CAM activation was not captured."

            )


        if not gradients:

            raise RuntimeError(

                "Grad-CAM gradient was not captured."

            )


        # ----------------------------------------------------------------------
        # Activation + Gradient
        # ----------------------------------------------------------------------

        activation = activations[0]

        gradient = gradients[0]


        # ----------------------------------------------------------------------
        # Grad-CAM Weights
        # ----------------------------------------------------------------------

        weights = gradient.mean(

            dim=(2, 3),

            keepdim=True

        )


        # ----------------------------------------------------------------------
        # Weighted Activation
        # ----------------------------------------------------------------------

        cam = (

            weights
            *
            activation

        ).sum(

            dim=1

        )


        # ----------------------------------------------------------------------
        # ReLU
        # ----------------------------------------------------------------------

        cam = torch.relu(

            cam

        )


        # ----------------------------------------------------------------------
        # Convert To NumPy
        # ----------------------------------------------------------------------

        heatmap = (

            cam[0]
            .detach()
            .cpu()
            .numpy()

        )


        # ----------------------------------------------------------------------
        # Normalize
        # ----------------------------------------------------------------------

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


        # ----------------------------------------------------------------------
        # Resize To Original Image
        # ----------------------------------------------------------------------

        heatmap = cv2.resize(

            heatmap,

            (
                rgb_image.shape[1],
                rgb_image.shape[0]
            ),

            interpolation=cv2.INTER_LINEAR

        )


        # ----------------------------------------------------------------------
        # Final Range
        # ----------------------------------------------------------------------

        heatmap = np.clip(

            heatmap,

            0.0,
            1.0

        ).astype(

            np.float32

        )


        return heatmap


    finally:

        # ----------------------------------------------------------------------
        # Remove Hooks
        # ----------------------------------------------------------------------

        forward_handle.remove()

        backward_handle.remove()


# ==============================================================================
# Generate Full-Test Calibration Dataset
# ==============================================================================

def generate_calibration_dataset() -> Path:

    """
    Generate Grad-CAM affected-area measurements for the
    complete test-set XAI results.

    No 76-image restriction is applied.
    """

    # ==========================================================================
    # Locate Full-Test XAI Results
    # ==========================================================================

    xai_results_file = (

        find_xai_batch_results()

    )


    # ==========================================================================
    # Header
    # ==========================================================================

    print()

    print("=" * 80)

    print(

        "PHASE 6 - PART 3"

    )

    print(

        "FULL-TEST-SET SEVERITY CALIBRATION DATASET"

    )

    print("=" * 80)

    print()


    print(

        f"Source XAI Results : "
        f"{xai_results_file}"

    )


    # ==========================================================================
    # Load XAI Results
    # ==========================================================================

    xai_data = pd.read_csv(

        xai_results_file

    )


    if xai_data.empty:

        raise RuntimeError(

            "Full-test XAI results file is empty."

        )


    # ==========================================================================
    # Column Normalization
    # ==========================================================================

    column_map = {

        column.lower().strip():
        column

        for column in xai_data.columns

    }


    def get_column(*names):

        for name in names:

            key = (

                name.lower().strip()

            )


            if key in column_map:

                return column_map[key]


        return None


    image_column = get_column(

        "Image",
        "Image_Name",
        "Filename",
        "File",
        "image_name"

    )


    image_path_column = get_column(

        "image_path",
        "Image_Path",
        "Path",
        "File_Path"

    )


    actual_column = get_column(

        "Actual",
        "Actual_Class",
        "True_Class",
        "True_Label",
        "actual",
        "actual_class"

    )


    predicted_column = get_column(

        "Predicted",
        "Predicted_Class",
        "Prediction",
        "Predicted_Label",
        "predicted",
        "predicted_class"

    )


    confidence_column = get_column(

        "Confidence",
        "Prediction_Confidence",
        "Confidence_Score",
        "confidence"

    )


    # ==========================================================================
    # Required Column Validation
    # ==========================================================================

    missing_columns = []


    if image_column is None:

        missing_columns.append(

            "Image"

        )


    if actual_column is None:

        missing_columns.append(

            "Actual"

        )


    if predicted_column is None:

        missing_columns.append(

            "Predicted"

        )


    if confidence_column is None:

        missing_columns.append(

            "Confidence"

        )


    if missing_columns:

        raise ValueError(

            "Phase 5 XAI results are missing required columns: "

            +

            ", ".join(

                missing_columns

            )

            +

            "\nAvailable columns: "

            +

            ", ".join(

                xai_data.columns

            )

        )


    # ==========================================================================
    # Load Grad-CAM Model Once
    # ==========================================================================

    print()

    print(

        "Loading EfficientNet-B0 for severity Grad-CAM..."

    )


    model, device = (

        load_severity_gradcam_model()

    )


    print(

        f"Grad-CAM Device     : "
        f"{device}"

    )


    print(

        "Grad-CAM Model       : "
        "EfficientNet-B0"

    )


    print()


    # ==========================================================================
    # Process All Images
    # ==========================================================================

    records = []

    successful_images = 0

    failed_images = 0


    total_images = len(

        xai_data

    )


    print(

        f"Images To Process   : "
        f"{total_images}"

    )


    print()

    print("=" * 80)

    print()


    for index, row in xai_data.iterrows():

        image_name = str(

            row[image_column]

        )


        print(

            f"[{index + 1}/{total_images}] "
            f"{image_name}"

        )


        # ----------------------------------------------------------------------
        # Locate Original Image
        # ----------------------------------------------------------------------

        original_path = None


        if image_path_column is not None:

            candidate_path = Path(

                str(

                    row[image_path_column]

                )

            )


            if candidate_path.exists():

                original_path = candidate_path


        # ----------------------------------------------------------------------
        # Fallback Search
        # ----------------------------------------------------------------------

        if original_path is None:

            matches = list(

                TEST_DIR.rglob(

                    image_name

                )

            )


            if matches:

                original_path = matches[0]


        # ----------------------------------------------------------------------
        # Image Not Found
        # ----------------------------------------------------------------------

        if original_path is None:

            failed_images += 1


            print(

                "    Status : FAILED"

            )


            print(

                "    Reason : Original image not found"

            )


            print()

            continue


        try:

            # ------------------------------------------------------------------
            # Generate Grad-CAM In Memory
            # ------------------------------------------------------------------

            heatmap = generate_severity_gradcam(

                model,

                device,

                original_path

            )


            # ------------------------------------------------------------------
            # Load Original RGB Image
            # ------------------------------------------------------------------

            image = load_rgb_image(

                original_path

            )


            # ------------------------------------------------------------------
            # Leaf Mask
            # ------------------------------------------------------------------

            leaf_mask = create_leaf_mask(

                image

            )


            # ------------------------------------------------------------------
            # Affected Region
            # ------------------------------------------------------------------

            (
                leaf_pixels,
                affected_pixels,
                affected_area_percent

            ) = calculate_affected_region(

                leaf_mask,

                heatmap

            )


            # ------------------------------------------------------------------
            # Store Result
            # ------------------------------------------------------------------

            records.append({

                "Image":

                    image_name,

                "Image_Path":

                    str(original_path),

                "Actual":

                    row[actual_column],

                "Predicted":

                    row[predicted_column],

                "Confidence":

                    row[confidence_column],

                "Leaf_Pixels":

                    leaf_pixels,

                "Affected_Pixels":

                    affected_pixels,

                "Affected_Area_Percent":

                    round(

                        affected_area_percent,

                        2

                    ),

                "Reference_Severity_Percent":

                    "",

                "Reference_Severity_Level":

                    "",

                "Annotation_Source":

                    "",

                "Annotation_Notes":

                    ""

            })


            successful_images += 1


            # ------------------------------------------------------------------
            # Console Result
            # ------------------------------------------------------------------

            print(

                f"    Actual          : "
                f"{row[actual_column]}"

            )


            print(

                f"    Predicted       : "
                f"{row[predicted_column]}"

            )


            print(

                f"    Confidence      : "
                f"{float(row[confidence_column]):.4f}"

            )


            print(

                f"    Leaf Pixels     : "
                f"{leaf_pixels}"

            )


            print(

                f"    Affected Pixels : "
                f"{affected_pixels}"

            )


            print(

                f"    Affected Area   : "
                f"{affected_area_percent:.2f}%"

            )


            print(

                "    Status          : SUCCESS"

            )


            print()


        except Exception as error:

            failed_images += 1


            print(

                "    Status          : FAILED"

            )


            print(

                f"    Reason          : "
                f"{error}"

            )


            print()


    # ==========================================================================
    # Validate Successful Results
    # ==========================================================================

    if not records:

        raise RuntimeError(

            "No severity measurements were generated."

        )


    # ==========================================================================
    # Create Calibration DataFrame
    # ==========================================================================

    calibration_data = pd.DataFrame(

        records

    )


    # ==========================================================================
    # Save Calibration File
    # ==========================================================================

    calibration_data.to_csv(

        CALIBRATION_FILE,

        index=False

    )


    # ==========================================================================
    # Coverage
    # ==========================================================================

    coverage_percent = (

        successful_images
        /
        total_images

    ) * 100.0


    # ==========================================================================
    # Completion Summary
    # ==========================================================================

    print()

    print("=" * 80)

    print(

        "PHASE 6 - PART 3 COMPLETED"

    )

    print("=" * 80)

    print()


    print(

        f"Images Evaluated    : "
        f"{total_images}"

    )


    print(

        f"Successful Images   : "
        f"{successful_images}"

    )


    print(

        f"Failed Images       : "
        f"{failed_images}"

    )


    print(

        f"Coverage            : "
        f"{coverage_percent:.2f}%"

    )


    print()


    print(

        f"Calibration File    : "
        f"{CALIBRATION_FILE}"

    )


    print()


    print(

        "Reference severity fields are intentionally blank."

    )


    print(

        "Independent expert annotation is required "

        "before severity accuracy can be validated."

    )


    print()


    print(

        "IMPORTANT:"

    )


    print(

        "Affected area is a Grad-CAM-derived severity proxy, "

        "not independently validated disease severity."

    )


    print()

    print("=" * 80)

    print()


    return CALIBRATION_FILE


# ==============================================================================
# End of Phase 6 - Part 3
# ==============================================================================
# ==============================================================================
# ==============================================================================
#
# PlantXAI-Net
#
# Phase 6
#
# Severity Estimation
#
# Part 4
#
# Reference Severity Validation & Threshold Calibration
#
# ==============================================================================


# ==============================================================================
# Calibration Report
# ==============================================================================

CALIBRATION_REPORT_FILE = (

    CALIBRATION_DIR

    / "severity_calibration_report.csv"

)


# ==============================================================================
# Validate Reference Annotations
# ==============================================================================

def validate_reference_annotations(

    calibration_file: Path

) -> pd.DataFrame:

    """
    Validate reference severity annotations.

    Reference severity must be independently assigned.
    No automatic severity labels are generated here.
    """

    data = pd.read_csv(

        calibration_file

    )


    # --------------------------------------------------------------------------
    # Required Columns
    # --------------------------------------------------------------------------

    required_columns = [

        "Image",

        "Actual",

        "Predicted",

        "Confidence",

        "Affected_Area_Percent",

        "Reference_Severity_Percent",

        "Reference_Severity_Level",

        "Annotation_Source",

        "Annotation_Notes"

    ]


    missing_columns = [

        column

        for column in required_columns

        if column not in data.columns

    ]


    if missing_columns:

        raise ValueError(

            "Calibration CSV is missing required columns: "

            +

            ", ".join(missing_columns)

        )


    # --------------------------------------------------------------------------
    # Normalize Reference Severity Level
    # --------------------------------------------------------------------------

    data["Reference_Severity_Level"] = (

        data["Reference_Severity_Level"]

        .fillna("")

        .astype(str)

        .str.strip()

    )


    # --------------------------------------------------------------------------
    # Convert Reference Severity Percentage
    # --------------------------------------------------------------------------

    data["Reference_Severity_Percent"] = pd.to_numeric(

        data["Reference_Severity_Percent"],

        errors="coerce"

    )


    # --------------------------------------------------------------------------
    # Validate Severity Levels
    # --------------------------------------------------------------------------

    valid_levels = set(

        CALIBRATION_LEVELS

    )


    invalid_levels = [

        level

        for level in data["Reference_Severity_Level"]

        if level

        and level not in valid_levels

    ]


    if invalid_levels:

        raise ValueError(

            "Invalid reference severity level(s): "

            +

            ", ".join(

                sorted(

                    set(invalid_levels)

                )

            )

            +

            ". Allowed levels: "

            +

            ", ".join(CALIBRATION_LEVELS)

        )


    # --------------------------------------------------------------------------
    # Validate Percentage Range
    # --------------------------------------------------------------------------

    invalid_percentages = data[

        data["Reference_Severity_Percent"].notna()

        &

        (

            (

                data["Reference_Severity_Percent"]

                < 0

            )

            |

            (

                data["Reference_Severity_Percent"]

                > 100

            )

        )

    ]


    if not invalid_percentages.empty:

        raise ValueError(

            "Reference severity percentage must be "

            "between 0 and 100."

        )


    # --------------------------------------------------------------------------
    # Check Annotation Completeness
    # --------------------------------------------------------------------------

    level_complete = (

        data["Reference_Severity_Level"]

        != ""

    )


    percentage_complete = (

        data["Reference_Severity_Percent"]

        .notna()

    )


    source_complete = (

        data["Annotation_Source"]

        .fillna("")

        .astype(str)

        .str.strip()

        != ""

    )


    data["Annotation_Complete"] = (

        level_complete

        &

        percentage_complete

        &

        source_complete

    )


    return data


# ==============================================================================
# Calculate Calibration Statistics
# ==============================================================================

def calculate_calibration_statistics(

    data: pd.DataFrame

) -> pd.DataFrame:

    """
    Calculate statistics for independently annotated
    severity samples.
    """

    annotated = data[

        data["Annotation_Complete"]

    ].copy()


    if annotated.empty:

        return pd.DataFrame()


    statistics = []


    for level in CALIBRATION_LEVELS:

        subset = annotated[

            annotated["Reference_Severity_Level"]

            == level

        ]


        if subset.empty:

            continue


        statistics.append({

            "Severity_Level":

                level,

            "Sample_Count":

                len(subset),

            "Reference_Mean_Percent":

                round(

                    subset[

                        "Reference_Severity_Percent"

                    ].mean(),

                    2

                ),

            "Reference_Min_Percent":

                round(

                    subset[

                        "Reference_Severity_Percent"

                    ].min(),

                    2

                ),

            "Reference_Max_Percent":

                round(

                    subset[

                        "Reference_Severity_Percent"

                    ].max(),

                    2

                ),

            "Affected_Area_Mean_Percent":

                round(

                    subset[

                        "Affected_Area_Percent"

                    ].mean(),

                    2

                ),

            "Affected_Area_Min_Percent":

                round(

                    subset[

                        "Affected_Area_Percent"

                    ].min(),

                    2

                ),

            "Affected_Area_Max_Percent":

                round(

                    subset[

                        "Affected_Area_Percent"

                    ].max(),

                    2

                )

        })


    return pd.DataFrame(

        statistics

    )


# ==============================================================================
# Calibration Readiness
# ==============================================================================

def analyze_calibration_readiness(

    data: pd.DataFrame

) -> None:

    """
    Display the current calibration status.

    Thresholds are not generated until reference
    severity annotations are available.
    """

    total_images = len(data)


    annotated_images = int(

        data["Annotation_Complete"].sum()

    )


    pending_images = (

        total_images

        -

        annotated_images

    )


    print()

    print("=" * 80)

    print("PHASE 6 - PART 4")

    print("REFERENCE SEVERITY VALIDATION")

    print("=" * 80)

    print()


    print(

        f"Total Calibration Images : "

        f"{total_images}"

    )


    print(

        f"Annotated Images         : "

        f"{annotated_images}"

    )


    print(

        f"Pending Annotations      : "

        f"{pending_images}"

    )


    print()


    if pending_images > 0:

        print(

            "Calibration Status       : PENDING"

        )

        print()

        print(

            "Reference severity annotations "

            "are required before threshold calibration."

        )

        print()

        print(

            "No automatic Mild / Moderate / Severe "

            "thresholds will be generated."

        )

    else:

        print(

            "Calibration Status       : READY"

        )

        print()

        print(

            "All calibration samples contain "

            "reference severity information."

        )


    print()

    print("=" * 80)

    print()


# ==============================================================================
# Generate Calibration Report
# ==============================================================================

def generate_calibration_report() -> Path:

    """
    Validate the calibration dataset and generate
    calibration statistics.
    """

    if not CALIBRATION_FILE.exists():

        raise FileNotFoundError(

            f"Calibration file not found: "

            f"{CALIBRATION_FILE}"

        )


    data = validate_reference_annotations(

        CALIBRATION_FILE

    )


    statistics = calculate_calibration_statistics(

        data

    )


    if statistics.empty:

        statistics = pd.DataFrame({

            "Severity_Level": [],

            "Sample_Count": [],

            "Reference_Mean_Percent": [],

            "Reference_Min_Percent": [],

            "Reference_Max_Percent": [],

            "Affected_Area_Mean_Percent": [],

            "Affected_Area_Min_Percent": [],

            "Affected_Area_Max_Percent": []

        })


    statistics.to_csv(

        CALIBRATION_REPORT_FILE,

        index=False

    )


    analyze_calibration_readiness(

        data

    )


    print(

        f"Calibration Report : "

        f"{CALIBRATION_REPORT_FILE}"

    )


    print()


    return CALIBRATION_REPORT_FILE


# ==============================================================================
# End of Phase 6 - Part 4
# ==============================================================================

# ==============================================================================
#
# PlantXAI-Net
#
# Phase 6
#
# Severity Estimation
#
# Part 5
#
# Severity Threshold Calibration
#
# ==============================================================================


# ==============================================================================
# Threshold Output
# ==============================================================================

SEVERITY_THRESHOLD_FILE = (

    CALIBRATION_DIR

    / "severity_thresholds.csv"

)


# ==============================================================================
# Calculate Data-Driven Thresholds
# ==============================================================================

def calculate_severity_thresholds(

    data: pd.DataFrame

) -> pd.DataFrame:

    """
    Calculate severity thresholds from independently
    annotated calibration samples.

    Thresholds are generated only when all three
    severity levels contain reference annotations.
    """

    annotated = data[

        data["Annotation_Complete"]

    ].copy()


    required_levels = [

        "Mild",

        "Moderate",

        "Severe"

    ]


    available_levels = set(

        annotated["Reference_Severity_Level"]

    )


    missing_levels = [

        level

        for level in required_levels

        if level not in available_levels

    ]


    if missing_levels:

        print()

        print("=" * 80)

        print("SEVERITY THRESHOLD CALIBRATION")

        print("=" * 80)

        print()

        print(

            "Calibration cannot be performed yet."

        )

        print()

        print(

            "Missing reference severity levels: "

            +

            ", ".join(missing_levels)

        )

        print()

        print(

            "Thresholds remain PENDING."

        )

        print()

        print("=" * 80)

        print()

        return pd.DataFrame()


    # --------------------------------------------------------------------------
    # Calculate class statistics
    # --------------------------------------------------------------------------

    statistics = []


    for level in required_levels:

        subset = annotated[

            annotated["Reference_Severity_Level"]

            == level

        ]


        affected_area = (

            subset["Affected_Area_Percent"]

            .astype(float)

        )


        statistics.append({

            "Severity_Level":

                level,

            "Sample_Count":

                len(subset),

            "Affected_Area_Mean":

                round(

                    affected_area.mean(),

                    2

                ),

            "Affected_Area_Min":

                round(

                    affected_area.min(),

                    2

                ),

            "Affected_Area_Max":

                round(

                    affected_area.max(),

                    2

                )

        })


    threshold_data = pd.DataFrame(

        statistics

    )


    # --------------------------------------------------------------------------
    # Check class ordering
    # --------------------------------------------------------------------------

    mild_max = threshold_data.loc[

        threshold_data["Severity_Level"]

        == "Mild",

        "Affected_Area_Max"

    ].iloc[0]


    moderate_min = threshold_data.loc[

        threshold_data["Severity_Level"]

        == "Moderate",

        "Affected_Area_Min"

    ].iloc[0]


    moderate_max = threshold_data.loc[

        threshold_data["Severity_Level"]

        == "Moderate",

        "Affected_Area_Max"

    ].iloc[0]


    severe_min = threshold_data.loc[

        threshold_data["Severity_Level"]

        == "Severe",

        "Affected_Area_Min"

    ].iloc[0]


    # --------------------------------------------------------------------------
    # Derive conservative boundaries
    # --------------------------------------------------------------------------

    mild_moderate_threshold = (

        mild_max + moderate_min

    ) / 2.0


    moderate_severe_threshold = (

        moderate_max + severe_min

    ) / 2.0


    final_thresholds = pd.DataFrame({

        "Threshold":

        [

            "Mild_to_Moderate",

            "Moderate_to_Severe"

        ],

        "Affected_Area_Percent":

        [

            round(

                mild_moderate_threshold,

                2

            ),

            round(

                moderate_severe_threshold,

                2

            )

        ]

    })


    # --------------------------------------------------------------------------
    # Save thresholds
    # --------------------------------------------------------------------------

    final_thresholds.to_csv(

        SEVERITY_THRESHOLD_FILE,

        index=False

    )


    print()

    print("=" * 80)

    print("SEVERITY THRESHOLD CALIBRATION")

    print("=" * 80)

    print()

    print(

        f"Mild → Moderate : "
        f"{mild_moderate_threshold:.2f}%"

    )

    print(

        f"Moderate → Severe : "
        f"{moderate_severe_threshold:.2f}%"

    )

    print()

    print(

        f"Threshold File : "
        f"{SEVERITY_THRESHOLD_FILE}"

    )

    print()

    print(

        "Calibration Status : COMPLETE"

    )

    print()

    print("=" * 80)

    print()


    return final_thresholds


# ==============================================================================
# Phase 6 Part 5 Analysis
# ==============================================================================

def run_severity_threshold_calibration() -> None:

    """
    Run the complete severity threshold calibration stage.
    """

    if not CALIBRATION_FILE.exists():

        raise FileNotFoundError(

            f"Calibration file not found: "

            f"{CALIBRATION_FILE}"

        )


    data = validate_reference_annotations(

        CALIBRATION_FILE

    )


    # --------------------------------------------------------------------------
    # Display calibration dataset summary
    # --------------------------------------------------------------------------

    print()

    print("=" * 80)

    print("PHASE 6 - PART 5")

    print("SEVERITY THRESHOLD CALIBRATION")

    print("=" * 80)

    print()

    print(

        f"Calibration Samples : {len(data)}"

    )

    print(

        f"Annotated Samples   : "

        f"{int(data['Annotation_Complete'].sum())}"

    )

    print()


    # --------------------------------------------------------------------------
    # Calculate thresholds
    # --------------------------------------------------------------------------

    thresholds = calculate_severity_thresholds(

        data

    )


    if thresholds.empty:

        print(

            "Threshold calibration is still pending."

        )

        print()

        print(

            "Add independent reference annotations "

            "for Mild, Moderate and Severe samples "

            "before running final calibration."

        )

        print()

        return


    print(

        "Phase 6 - Part 5 Completed Successfully"

    )

    print()


# ==============================================================================
# End of Phase 6 - Part 5
# ==============================================================================

# ==============================================================================
#
# PlantXAI-Net
#
# Phase 6
#
# Severity Estimation
#
# Part 6
#
# Final Grad-CAM Severity Estimation
#
# ==============================================================================


# ==============================================================================
# Provisional Severity Thresholds
# ==============================================================================

MILD_THRESHOLD = 20.0

MODERATE_THRESHOLD = 50.0


# ==============================================================================
# Severity Classification
# ==============================================================================

def classify_severity(

    affected_area_percent: float

) -> str:

    """
    Convert Grad-CAM affected-area percentage into
    a provisional severity category.

    This is a Grad-CAM-derived severity proxy and
    not an independently validated ground-truth severity.
    """

    if affected_area_percent < MILD_THRESHOLD:

        return "Mild"

    elif affected_area_percent < MODERATE_THRESHOLD:

        return "Moderate"

    else:

        return "Severe"


# ==============================================================================
# Generate Final Severity Results
# ==============================================================================

def generate_final_severity_results() -> Path:

    """
    Generate final severity proxy results from the
    calibrated Grad-CAM affected-area measurements.
    """

    if not CALIBRATION_FILE.exists():

        raise FileNotFoundError(

            f"Calibration file not found: "
            f"{CALIBRATION_FILE}"

        )


    data = pd.read_csv(

        CALIBRATION_FILE

    )


    required_columns = [

        "Image",

        "Actual",

        "Predicted",

        "Confidence",

        "Affected_Area_Percent"

    ]


    missing_columns = [

        column

        for column in required_columns

        if column not in data.columns

    ]


    if missing_columns:

        raise ValueError(

            "Required columns missing: "

            +

            ", ".join(missing_columns)

        )


    # --------------------------------------------------------------------------
    # Calculate Severity Proxy
    # --------------------------------------------------------------------------

    data["Severity_Percent"] = (

        pd.to_numeric(

            data["Affected_Area_Percent"],

            errors="coerce"

        )

        .clip(

            lower=0,

            upper=100

        )

    )


    data["Severity_Level"] = (

        data["Severity_Percent"]

        .apply(

            classify_severity

        )

    )


    # --------------------------------------------------------------------------
    # Save Final Results
    # --------------------------------------------------------------------------

    final_file = (

        SEVERITY_RESULTS_DIR

        / "final_severity_results.csv"

    )


    data.to_csv(

        final_file,

        index=False

    )


    # --------------------------------------------------------------------------
    # Display Results
    # --------------------------------------------------------------------------

    print()

    print("=" * 80)

    print("PHASE 6 - PART 6")

    print("FINAL GRAD-CAM SEVERITY ESTIMATION")

    print("=" * 80)

    print()


    print(

        f"Images Processed : {len(data)}"

    )


    print(

        f"Mild             : "

        f"{(data['Severity_Level'] == 'Mild').sum()}"

    )


    print(

        f"Moderate         : "

        f"{(data['Severity_Level'] == 'Moderate').sum()}"

    )


    print(

        f"Severe           : "

        f"{(data['Severity_Level'] == 'Severe').sum()}"

    )


    print()


    print(

        "Severity Thresholds:"

    )

    print(

        f"  Mild      : < {MILD_THRESHOLD:.1f}%"

    )

    print(

        f"  Moderate  : "

        f"{MILD_THRESHOLD:.1f}% - "
        f"{MODERATE_THRESHOLD:.1f}%"

    )

    print(

        f"  Severe    : >= {MODERATE_THRESHOLD:.1f}%"

    )


    print()


    print(

        f"Results Saved : {final_file}"

    )


    print()


    print(

        "NOTE: Severity is reported as a "

        "Grad-CAM-derived proxy."

    )

    print(

        "Independent severity ground truth is "

        "required for accuracy validation."

    )


    print()

    print("=" * 80)

    print(

        "Phase 6 - Part 6 Completed Successfully"

    )

    print("=" * 80)

    print()


    return final_file


# ==============================================================================
# End of Phase 6 - Part 6
# ==============================================================================

# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":

    # --------------------------------------------------------------------------
    # Phase 6 - Part 1
    # --------------------------------------------------------------------------

    verify_severity_configuration()


    # --------------------------------------------------------------------------
    # Phase 6 - Part 3
    # Generate fresh calibration data from latest XAI results
    # --------------------------------------------------------------------------

    generate_calibration_dataset()


    # --------------------------------------------------------------------------
    # Phase 6 - Part 6
    # Generate final Grad-CAM severity results
    # --------------------------------------------------------------------------

    generate_final_severity_results()