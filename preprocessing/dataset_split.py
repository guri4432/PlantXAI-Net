import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

# =====================================================
# Configuration
# =====================================================

SOURCE_DIR = Path(r"D:\PlantXAI-Net\dataset\PlantVillage\color")
SPLIT_DIR = Path(r"D:\PlantXAI-Net\dataset\PlantVillage\split")

TRAIN_DIR = SPLIT_DIR / "train"
VAL_DIR = SPLIT_DIR / "validation"
TEST_DIR = SPLIT_DIR / "test"

RANDOM_SEED = 42

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

# Create folders
for folder in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("Creating Train / Validation / Test Split")
print("=" * 60)

summary = []

# =====================================================
# Process each class
# =====================================================

for class_name in sorted(os.listdir(SOURCE_DIR)):

    class_path = SOURCE_DIR / class_name

    if not class_path.is_dir():
        continue

    images = [
        img for img in class_path.iterdir()
        if img.suffix.lower() in IMAGE_EXTENSIONS
    ]

    # 80% Train
    train_imgs, temp_imgs = train_test_split(
        images,
        test_size=0.20,
        random_state=RANDOM_SEED,
        shuffle=True
    )

    # Remaining 20% → 10% Validation + 10% Test
    val_imgs, test_imgs = train_test_split(
        temp_imgs,
        test_size=0.50,
        random_state=RANDOM_SEED,
        shuffle=True
    )

    # Create destination folders
    (TRAIN_DIR / class_name).mkdir(exist_ok=True)
    (VAL_DIR / class_name).mkdir(exist_ok=True)
    (TEST_DIR / class_name).mkdir(exist_ok=True)

    # Copy files
    for img in train_imgs:
        shutil.copy2(img, TRAIN_DIR / class_name / img.name)

    for img in val_imgs:
        shutil.copy2(img, VAL_DIR / class_name / img.name)

    for img in test_imgs:
        shutil.copy2(img, TEST_DIR / class_name / img.name)

    summary.append([
        class_name,
        len(images),
        len(train_imgs),
        len(val_imgs),
        len(test_imgs)
    ])

# =====================================================
# Print Summary
# =====================================================

print("\nSplit Completed Successfully!\n")

print(f"{'Class':50} {'Total':>6} {'Train':>6} {'Val':>6} {'Test':>6}")

print("-" * 80)

for row in summary:
    print(f"{row[0]:50} {row[1]:6} {row[2]:6} {row[3]:6} {row[4]:6}")

print("=" * 80)

print("Dataset Split Completed.")