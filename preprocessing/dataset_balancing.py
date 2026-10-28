# ==============================================================================
#                           PlantXAI-Net
# ==============================================================================
#
# Module      : Dataset Balancing
# Phase       : 1
# Version     : 2.0
#
# Title
# -----
# Imports, Configuration, Logging & Augmentation Pipeline
#
# Description
# -----------
# This phase initializes all required libraries, project configuration,
# logging system, random seed, supported image extensions and the
# Albumentations augmentation pipeline.
#
# Audit Improvements (Version 2.0)
# --------------------------------
# ✔ Added missing NumPy import
# ✔ Removed unused imports
# ✔ Standardized configuration names
# ✔ Improved readability
# ✔ Compatible with Phases 2–4
#
# ==============================================================================

# ==============================================================================
# Imports
# ==============================================================================

import cv2
import random
import shutil
import logging

from pathlib import Path

import numpy as np
from tqdm import tqdm
import albumentations as A

# ==============================================================================
# Project Configuration
# ==============================================================================

# Project Root Directory

PROJECT_ROOT = Path(r"D:\PlantXAI-Net")

# ------------------------------------------------------------------------------
# Dataset Paths
# ------------------------------------------------------------------------------

INPUT_DATASET = (
    PROJECT_ROOT
    / "dataset"
    / "PlantVillage"
    / "split"
    / "train"
)

OUTPUT_DATASET = (
    PROJECT_ROOT
    / "dataset"
    / "PlantVillage"
    / "balanced"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "results"
    / "reports"
)

# ------------------------------------------------------------------------------
# Dataset Configuration
# ------------------------------------------------------------------------------

TARGET_IMAGES_PER_CLASS = 1500

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png"
)

SEED = 42

# ==============================================================================
# Random Seed
# ==============================================================================

random.seed(SEED)

# ==============================================================================
# Create Required Directories
# ==============================================================================

OUTPUT_DATASET.mkdir(
    parents=True,
    exist_ok=True
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ==============================================================================
# Logging Configuration
# ==============================================================================

LOG_FILE = REPORT_DIR / "balancing_log.txt"

logging.basicConfig(

    level=logging.INFO,

    format="%(asctime)s | %(levelname)s | %(message)s",

    handlers=[

        logging.FileHandler(LOG_FILE),

        logging.StreamHandler()

    ]

)

logger = logging.getLogger(__name__)


# ==============================================================================
# Albumentations Augmentation Pipeline
# ==============================================================================

augmentation = A.Compose([

    A.HorizontalFlip(
        p=0.5
    ),

    A.Rotate(
        limit=15,
        p=0.4
    ),

    A.RandomBrightnessContrast(

        brightness_limit=0.15,

        contrast_limit=0.15,

        p=0.4

    ),

    A.ShiftScaleRotate(

        shift_limit=0.05,

        scale_limit=0.05,

        rotate_limit=10,

        border_mode=cv2.BORDER_REFLECT,

        p=0.35

    ),

    A.RandomGamma(

        gamma_limit=(90, 110),

        p=0.3

    ),

    A.GaussNoise(

        std_range=(0.02, 0.08),

        p=0.2

    )

])

# ==============================================================================
# Utility Information
# ==============================================================================

logger.info("=" * 80)
logger.info("PlantXAI-Net Dataset Balancing")
logger.info("Phase 1 (Version 2.0) Initialized Successfully")
logger.info(f"Input Dataset        : {INPUT_DATASET}")
logger.info(f"Output Dataset       : {OUTPUT_DATASET}")
logger.info(f"Reports Directory    : {REPORT_DIR}")
logger.info(f"Target Images/Class  : {TARGET_IMAGES_PER_CLASS}")
logger.info(f"Random Seed          : {SEED}")
logger.info("=" * 80)

# ==============================================================================
#                           END OF PHASE 1
# ==============================================================================



# ==============================================================================
#                           PlantXAI-Net
# ==============================================================================
#
# Module      : Dataset Balancing
# Phase       : 2
# Version     : 3.1
#
# Title
# -----
# Helper Functions
#
# Description
# -----------
# This phase contains helper functions required for dataset balancing,
# including image discovery, image reading, validation, image saving,
# augmentation support and safe filename generation.
#
# Audit Improvements (Version 3.1)
# --------------------------------
# ✔ Added type hints
# ✔ Improved exception handling
# ✔ Improved documentation
# ✔ Better runtime safety
# ✔ Fully compatible with Phase 3
#
# ==============================================================================


def get_image_files(folder: Path) -> list[Path]:
    """
    Retrieve all valid image files from a directory.

    Parameters
    ----------
    folder : Path
        Directory containing images.

    Returns
    -------
    list[Path]
        Sorted list of image paths.

    Raises
    ------
    FileNotFoundError
        If folder does not exist.
    """

    if not folder.exists():
        raise FileNotFoundError(
            f"Folder not found: {folder}"
        )

    image_files = sorted(
        [
            file
            for file in folder.iterdir()
            if file.is_file()
            and file.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )

    return image_files


# ------------------------------------------------------------------------------


def read_image(image_path: Path) -> np.ndarray:
    """
    Safely read an RGB image.

    Parameters
    ----------
    image_path : Path

    Returns
    -------
    np.ndarray
        RGB image.

    Raises
    ------
    FileNotFoundError
        If image does not exist.

    ValueError
        If OpenCV fails to read the image.
    """

    if not image_path.exists():

        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    image = cv2.imread(str(image_path))

    if image is None:

        raise ValueError(
            f"Unable to read image: {image_path}"
        )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    return image


# ------------------------------------------------------------------------------


def validate_image(image: np.ndarray) -> bool:
    """
    Validate an image before processing.

    Parameters
    ----------
    image : np.ndarray

    Returns
    -------
    bool

    Raises
    ------
    TypeError
    ValueError
    """

    if image is None:

        raise ValueError(
            "Image is None."
        )

    if not isinstance(image, np.ndarray):

        raise TypeError(
            "Image must be a NumPy array."
        )

    if image.size == 0:

        raise ValueError(
            "Image is empty."
        )

    if image.ndim != 3:

        raise ValueError(
            "Image must have exactly 3 dimensions."
        )

    if image.shape[2] != 3:

        raise ValueError(
            "Image must contain exactly 3 channels."
        )

    return True


# ------------------------------------------------------------------------------
#
# Safe Image Saving
#
# ------------------------------------------------------------------------------


def save_image(
    image: np.ndarray,
    save_path: Path
) -> bool:
    """
    Safely save an RGB image.

    Features
    --------
    ✔ Image validation
    ✔ Automatic folder creation
    ✔ RGB → BGR conversion
    ✔ Save verification

    Parameters
    ----------
    image : np.ndarray

    save_path : Path

    Returns
    -------
    bool
    """

    try:

        validate_image(image)

        save_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        image_bgr = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2BGR
        )

        success = cv2.imwrite(
            str(save_path),
            image_bgr
        )

        if not success:

            raise IOError(
                f"OpenCV failed to save image: {save_path}"
            )

        if not save_path.exists():

            raise FileNotFoundError(
                f"Image not found after saving: {save_path}"
            )

        if save_path.stat().st_size == 0:

            raise IOError(
                f"Saved image is empty: {save_path}"
            )

        logger.debug(
            f"Saved image: {save_path.name}"
        )

        return True

    except Exception as error:

        logger.error(
            f"Image Save Error ({save_path.name}) : {error}"
        )

        raise

# ------------------------------------------------------------------------------
#
# Copy Original Images
#
# ------------------------------------------------------------------------------

def copy_original_images(
    source_folder: Path,
    destination_folder: Path
) -> int:
    """
    Copy original images into the balanced dataset.

    Existing files are skipped.

    Parameters
    ----------
    source_folder : Path

    destination_folder : Path

    Returns
    -------
    int
        Number of successfully copied images.
    """

    destination_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    image_paths = get_image_files(source_folder)

    copied = 0
    skipped = 0
    failed = 0

    for image_path in image_paths:

        destination = destination_folder / image_path.name

        if destination.exists():

            skipped += 1
            continue

        try:

            shutil.copy2(
                image_path,
                destination
            )

            copied += 1

        except Exception as error:

            failed += 1

            logger.error(
                f"Copy Failed ({image_path.name}) : {error}"
            )

    logger.info(
        f"Copy Summary | "
        f"Copied: {copied} | "
        f"Skipped: {skipped} | "
        f"Failed: {failed}"
    )

    # NOTE:
    # Returning only 'copied' to maintain compatibility
    # with Phase 3 Version 1.0.
    return copied


# ------------------------------------------------------------------------------
#
# Image Augmentation
#
# ------------------------------------------------------------------------------


def augment_single_image(
    image: np.ndarray
) -> np.ndarray:
    """
    Apply the Albumentations pipeline to a single RGB image.

    Parameters
    ----------
    image : np.ndarray
        RGB image.

    Returns
    -------
    np.ndarray
        Augmented RGB image.
    """

    try:

        validate_image(image)

        transformed = augmentation(
            image=image
        )

        augmented = transformed["image"]

        validate_image(
            augmented
        )

        return augmented

    except Exception as error:

        logger.error(
            f"Augmentation Failed : {error}"
        )

        raise

# ------------------------------------------------------------------------------
#
# Resume Support
#
# ------------------------------------------------------------------------------

def get_next_augmented_index(
    destination_folder: Path
) -> int:
    """
    Find the next available augmentation index.

    Parameters
    ----------
    destination_folder : Path

    Returns
    -------
    int
        Next available augmentation index.
    """

    destination_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    highest_index = 0

    for file in destination_folder.iterdir():

        if not file.is_file():
            continue

        stem = file.stem

        if "_aug_" not in stem:
            continue

        try:

            index = int(
                stem.split("_aug_")[-1]
            )

            highest_index = max(
                highest_index,
                index
            )

        except ValueError:

            logger.warning(
                f"Ignoring invalid filename: {file.name}"
            )

            continue

    return highest_index + 1


# ------------------------------------------------------------------------------
#
# Safe Filename Generator
#
# ------------------------------------------------------------------------------

def generate_unique_filename(
    image_path: Path,
    destination_folder: Path,
    next_index: int
) -> tuple[str, int]:
    """
    Generate a unique filename for an augmented image.

    Parameters
    ----------
    image_path : Path

    destination_folder : Path

    next_index : int

    Returns
    -------
    tuple[str, int]
        (filename, updated_index)
    """

    while True:

        filename = (
            f"{image_path.stem}"
            f"_aug_{next_index:06d}.jpg"
        )

        save_path = destination_folder / filename

        if not save_path.exists():

            return filename, next_index

        next_index += 1


# ------------------------------------------------------------------------------
#
# Generate Augmented Images
#
# ------------------------------------------------------------------------------

def generate_augmented_images(
    image_paths: list[Path],
    destination_folder: Path,
    required_count: int
) -> int:
    """
    Generate augmented images until the required count is reached.

    Features
    --------
    ✔ Resume support
    ✔ Retry protection
    ✔ Safe progress bar cleanup
    ✔ No file overwrite
    ✔ Image validation

    Returns
    -------
    int
        Number of successfully generated images.
    """

    if required_count <= 0:

        logger.info(
            "No augmented images required."
        )

        return 0

    if not image_paths:

        raise ValueError(
            "No source images available for augmentation."
        )

    destination_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    generated = 0

    next_index = get_next_augmented_index(
        destination_folder
    )

    retry_count = 0
    max_retry_attempts = max(
        required_count * 10,
        100
    )

    progress = tqdm(
        total=required_count,
        desc="Generating Augmented Images",
        unit="img"
    )

    try:

        while generated < required_count:

            if retry_count >= max_retry_attempts:

                raise RuntimeError(
                    "Maximum retry limit reached during augmentation."
                )

            img_path = random.choice(
                image_paths
            )

            try:

                image = read_image(
                    img_path
                )

                augmented = augment_single_image(
                    image
                )

                filename, next_index = generate_unique_filename(
                    img_path,
                    destination_folder,
                    next_index
                )

                save_path = destination_folder / filename

                save_image(
                    augmented,
                    save_path
                )

                generated += 1
                next_index += 1

                retry_count = 0

                progress.update(1)

            except Exception as error:

                retry_count += 1

                logger.error(
                    f"Augmentation Failed ({img_path.name}) "
                    f"[Retry {retry_count}/{max_retry_attempts}] : {error}"
                )

    finally:

        progress.close()

    logger.info(
        f"Generated {generated} augmented images."
    )

    return generated

# ==============================================================================
# Phase 2 Audit Summary (Version 3.1)
# ==============================================================================
#
# Audit Improvements
# ------------------
# ✔ Added type hints to helper functions
# ✔ Improved documentation
# ✔ Improved exception handling
# ✔ Added retry protection for augmentation
# ✔ Eliminated infinite loop possibility
# ✔ Safe tqdm cleanup using try...finally
# ✔ Better logging messages
# ✔ Added failed copy statistics
# ✔ Improved filename generation
# ✔ Fully compatible with Phase 3 Version 1.0
#
# Runtime Status
# --------------
# ✔ No breaking changes
# ✔ Backward compatible
# ✔ Research-grade helper functions
#
# ==============================================================================
#                           END OF PHASE 2
# ==============================================================================

# ==============================================================================
#                           PlantXAI-Net
# ==============================================================================
#
# Module      : Dataset Balancing
# Phase       : 3
# Version     : 2.0
#
# Title
# -----
# Dataset Balancing Logic
#
# Description
# -----------
# This phase performs class-wise dataset balancing by copying original images,
# generating augmented images, verifying the final image count, and collecting
# statistics for each class.
#
# Audit Improvements (Version 2.0)
# --------------------------------
# ✔ Added type hints
# ✔ Introduced status constants
# ✔ Improved status handling
# ✔ Better documentation
# ✔ Fully compatible with Phase 2 Version 3.1
#
# ==============================================================================

# ==============================================================================
# Status Constants
# ==============================================================================

STATUS_BALANCED = "Balanced"
STATUS_COMPLETED = "Completed"
STATUS_EXCEEDED = "Exceeded"
STATUS_INCOMPLETE = "Incomplete"
STATUS_SKIPPED = "Skipped"
STATUS_FAILED = "Failed"


# ------------------------------------------------------------------------------
#
# Balance Single Class
#
# ------------------------------------------------------------------------------

def balance_single_class(
    class_name: str,
    source_folder: Path,
    destination_folder: Path
) -> dict:
    """
    Balance a single dataset class.

    Status Values
    -------------
    Balanced
        Final image count equals target.

    Completed
        Target already existed before augmentation.

    Exceeded
        Final count exceeds target.

    Incomplete
        Target could not be reached.

    Skipped
        No images available.

    Failed
        Unexpected processing error.

    Parameters
    ----------
    class_name : str

    source_folder : Path

    destination_folder : Path

    Returns
    -------
    dict
        Processing statistics.
    """

    logger.info("=" * 70)
    logger.info(f"Processing Class : {class_name}")
    logger.info("=" * 70)

    try:

        image_paths = get_image_files(
            source_folder
        )

        original_count = len(
            image_paths
        )

        if original_count == 0:

            logger.warning(
                f"No images found in class: {class_name}"
            )

            return {
                "class_name": class_name,
                "original": 0,
                "copied": 0,
                "augmented": 0,
                "final": 0,
                "status": STATUS_SKIPPED
            }

        logger.info(
            f"Original Images : {original_count}"
        )

        copied_count = copy_original_images(
            source_folder,
            destination_folder
        )

        current_count = len(
            get_image_files(destination_folder)
        )

        logger.info(
            f"Images After Copy : {current_count}"
        )

        required_augmentation = max(
            0,
            TARGET_IMAGES_PER_CLASS - current_count
        )

        logger.info(
            f"Required Augmentation : "
            f"{required_augmentation}"
        )

        if required_augmentation == 0:

            logger.info(
                "Target already achieved."
            )

            return {
                "class_name": class_name,
                "original": original_count,
                "copied": copied_count,
                "augmented": 0,
                "final": current_count,
                "status": STATUS_COMPLETED
            }

        logger.info(
            "Starting augmentation..."
        )

        augmented_count = generate_augmented_images(
            image_paths=image_paths,
            destination_folder=destination_folder,
            required_count=required_augmentation
        )

        final_count = len(
            get_image_files(destination_folder)
        )

        logger.info(
            f"Final Image Count : {final_count}"
        )

        if final_count == TARGET_IMAGES_PER_CLASS:

            status = STATUS_BALANCED

        elif final_count > TARGET_IMAGES_PER_CLASS:

            status = STATUS_EXCEEDED

        else:

            status = STATUS_INCOMPLETE

        logger.info(
            f"Processing Status : {status}"
        )

        return {

            "class_name": class_name,

            "original": original_count,

            "copied": copied_count,

            "augmented": augmented_count,

            "final": final_count,

            "status": status

        }

    except Exception as error:

        logger.exception(
            f"Failed to process class "
            f"{class_name} : {error}"
        )

        return {

            "class_name": class_name,

            "original": 0,

            "copied": 0,

            "augmented": 0,

            "final": 0,

            "status": STATUS_FAILED

        }
    
# ------------------------------------------------------------------------------
#
# Process Complete Dataset
#
# ------------------------------------------------------------------------------

def process_dataset() -> list[dict]:
    """
    Process every class in the input dataset.

    Each class is balanced independently. If processing of one class
    fails, the remaining classes continue processing.

    Returns
    -------
    list[dict]
        Statistics for every processed class.

    Raises
    ------
    FileNotFoundError
        If the input dataset directory does not exist.
    """

    logger.info("=" * 80)
    logger.info("Starting Dataset Balancing")
    logger.info("=" * 80)

    # ------------------------------------------------------------------
    # Validate Input Dataset
    # ------------------------------------------------------------------

    if not INPUT_DATASET.exists():

        raise FileNotFoundError(
            f"Input dataset not found: {INPUT_DATASET}"
        )

    OUTPUT_DATASET.mkdir(
        parents=True,
        exist_ok=True
    )

    # ------------------------------------------------------------------
    # Collect Class Folders
    # ------------------------------------------------------------------

    class_folders = sorted(
        [
            folder
            for folder in INPUT_DATASET.iterdir()
            if folder.is_dir()
        ]
    )

    logger.info(
        f"Total Classes Found : {len(class_folders)}"
    )

    statistics: list[dict] = []

    # ------------------------------------------------------------------
    # Process Every Class
    # ------------------------------------------------------------------

    for class_folder in class_folders:

        class_name = class_folder.name

        destination_folder = (
            OUTPUT_DATASET /
            class_name
        )

        try:

            result = balance_single_class(
                class_name=class_name,
                source_folder=class_folder,
                destination_folder=destination_folder
            )

            statistics.append(result)

        except Exception as error:

            logger.exception(
                f"Processing Failed ({class_name}) : {error}"
            )

            statistics.append(
                {
                    "class_name": class_name,
                    "original": 0,
                    "copied": 0,
                    "augmented": 0,
                    "final": 0,
                    "status": STATUS_FAILED
                }
            )

    logger.info("=" * 80)
    logger.info("Dataset Processing Completed")
    logger.info("=" * 80)

    return statistics

# ------------------------------------------------------------------------------
#
# Verify Balanced Dataset
#
# ------------------------------------------------------------------------------

def verify_dataset(
    statistics: list[dict]
) -> bool:
    """
    Verify the balanced dataset and display a detailed summary.

    Status Classification
    ---------------------
    Success :
        - Balanced
        - Completed

    Warning :
        - Exceeded
        - Skipped

    Failure :
        - Incomplete
        - Failed

    Parameters
    ----------
    statistics : list[dict]

    Returns
    -------
    bool
        True if no failures occurred.
    """

    logger.info("=" * 80)
    logger.info("Verifying Balanced Dataset")
    logger.info("=" * 80)

    total_original = 0
    total_copied = 0
    total_augmented = 0
    total_final = 0

    successful_classes = 0
    warning_classes = 0
    failed_classes = 0

    print()
    print("=" * 95)
    print("DATASET BALANCING SUMMARY")
    print("=" * 95)

    header = (
        f"{'Class':35}"
        f"{'Original':>10}"
        f"{'Copied':>10}"
        f"{'Augmented':>12}"
        f"{'Final':>10}"
        f"{'Status':>18}"
    )

    print(header)
    print("-" * 95)

    for item in statistics:

        total_original += item["original"]
        total_copied += item["copied"]
        total_augmented += item["augmented"]
        total_final += item["final"]

        status = item["status"]

        # ----------------------------------------------------------
        # Status Classification
        # ----------------------------------------------------------

        if status in (
            STATUS_BALANCED,
            STATUS_COMPLETED
        ):

            successful_classes += 1

        elif status in (
            STATUS_EXCEEDED,
            STATUS_SKIPPED
        ):

            warning_classes += 1

        else:

            failed_classes += 1

        print(

            f"{item['class_name'][:35]:35}"
            f"{item['original']:>10}"
            f"{item['copied']:>10}"
            f"{item['augmented']:>12}"
            f"{item['final']:>10}"
            f"{status:>18}"

        )

    print("-" * 95)

    print(
        f"{'TOTAL':35}"
        f"{total_original:>10}"
        f"{total_copied:>10}"
        f"{total_augmented:>12}"
        f"{total_final:>10}"
    )

    print("=" * 95)

    # ----------------------------------------------------------
    # Logging Summary
    # ----------------------------------------------------------

    logger.info(f"Classes Processed : {len(statistics)}")
    logger.info(f"Successful Classes : {successful_classes}")
    logger.info(f"Warning Classes : {warning_classes}")
    logger.info(f"Failed Classes : {failed_classes}")

    logger.info(f"Original Images : {total_original}")
    logger.info(f"Copied Images : {total_copied}")
    logger.info(f"Augmented Images : {total_augmented}")
    logger.info(f"Final Images : {total_final}")

    # ----------------------------------------------------------
    # Final Result
    # ----------------------------------------------------------

    if failed_classes == 0:

        logger.info(
            "Dataset verification completed successfully."
        )

        print()
        print("✅ Dataset Verification Successful.")
        print()

        if warning_classes > 0:

            print(
                f"⚠ {warning_classes} class(es) generated warning(s). "
                "Please review the summary above."
            )
            print()

        return True

    logger.error(
        "Dataset verification failed."
    )

    print()
    print(
        f"❌ Dataset Verification Failed "
        f"({failed_classes} class(es))."
    )
    print()

    return False

# ==============================================================================
# Phase 3 Audit Summary (Version 2.0)
# ==============================================================================
#
# Audit Improvements
# ------------------
# ✔ Added type hints
# ✔ Introduced reusable status constants
# ✔ Removed hard-coded status strings
# ✔ Improved exception handling
# ✔ Improved dataset verification logic
# ✔ Separated Success, Warning and Failure states
# ✔ Improved logging consistency
# ✔ Improved documentation
# ✔ Maintained backward compatibility with Phase 2
#
# Status Classification
# ---------------------
# Success
#   • Balanced
#   • Completed
#
# Warning
#   • Exceeded
#   • Skipped
#
# Failure
#   • Incomplete
#   • Failed
#
# Runtime Status
# --------------
# ✔ No breaking changes
# ✔ Compatible with Phase 1 Version 2.0
# ✔ Compatible with Phase 2 Version 3.1
# ✔ Ready for Phase 4
#
# ==============================================================================
#                           END OF PHASE 3
# ==============================================================================


# ==============================================================================
#                           PlantXAI-Net
# ==============================================================================
#
# Module      : Dataset Balancing
# Phase       : 4
# Version     : 1.1
#
# Title
# -----
# Main Execution
#
# Description
# -----------
# Controls execution of the dataset balancing pipeline,
# records execution time, verifies the balanced dataset
# and displays the final execution summary.
#
# Audit Improvements (Version 1.1)
# --------------------------------
# ✔ Better execution summary
# ✔ Human-readable execution time
# ✔ Improved logging
# ✔ Fully compatible with Phase 3 Version 2.0
#
# ==============================================================================

import time


def format_execution_time(seconds: float) -> str:
    """
    Convert execution time into a human-readable format.

    Parameters
    ----------
    seconds : float

    Returns
    -------
    str
    """

    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)

    if hours > 0:
        return f"{hours} hr {minutes} min {secs} sec"

    if minutes > 0:
        return f"{minutes} min {secs} sec"

    return f"{seconds:.2f} sec"


# ------------------------------------------------------------------------------


def main() -> None:
    """
    Main execution function.
    """

    print()
    print("=" * 80)
    print("PlantXAI-Net : Dataset Balancing")
    print("=" * 80)

    logger.info("Program Started")

    start_time = time.perf_counter()

    try:

        # ------------------------------------------------------
        # Process Dataset
        # ------------------------------------------------------

        statistics = process_dataset()

        # ------------------------------------------------------
        # Verify Dataset
        # ------------------------------------------------------

        verification_status = verify_dataset(
            statistics
        )

        # ------------------------------------------------------
        # Execution Time
        # ------------------------------------------------------

        execution_time = (
            time.perf_counter() - start_time
        )

        formatted_time = format_execution_time(
            execution_time
        )

        print("=" * 80)
        print(f"Execution Time : {formatted_time}")
        print("=" * 80)

        logger.info(
            f"Execution Time : {formatted_time}"
        )

        if verification_status:

            logger.info(
                "Pipeline Finished Successfully."
            )

            print("\n✅ Dataset balancing pipeline completed successfully.\n")

        else:

            logger.warning(
                "Pipeline Finished With Warnings."
            )

            print("\n⚠ Dataset balancing pipeline completed with warnings.\n")

    except Exception as error:

        logger.exception(
            f"Fatal Error : {error}"
        )

        print()
        print("=" * 80)
        print("Pipeline Execution Failed.")
        print("=" * 80)

        raise
    
# ------------------------------------------------------------------------------
# Program Entry Point
# ------------------------------------------------------------------------------

if __name__ == "__main__":

    main()


# ==============================================================================
# Phase 4 Audit Summary (Version 1.1)
# ==============================================================================
#
# Audit Improvements
# ------------------
# ✔ Improved main() structure
# ✔ Added human-readable execution time formatting
# ✔ Improved runtime logging
# ✔ Better success and warning reporting
# ✔ Added robust exception handling
# ✔ Added clear program entry point
# ✔ Maintained backward compatibility with previous phases
#
# Runtime Behaviour
# -----------------
# 1. Validate input dataset
# 2. Process every class
# 3. Copy original images
# 4. Generate augmented images
# 5. Verify balanced dataset
# 6. Display dataset summary
# 7. Display execution time
# 8. Exit gracefully
#
# Compatibility
# -------------
# ✔ Phase 1 Version 2.0
# ✔ Phase 2 Version 3.1
# ✔ Phase 3 Version 2.0
# ✔ Python 3.10+
#
# ==============================================================================
#                    DATASET BALANCING MODULE COMPLETED
# ==============================================================================
#
# Final Module Status
# -------------------
# Module Name : Dataset Balancing
# Final Version : 2.0
# Research Status : Production / Research Grade
#
# ==============================================================================
#                           END OF PHASE 4
# ==============================================================================