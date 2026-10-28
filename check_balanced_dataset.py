from pathlib import Path

BALANCED_DATASET = Path(r"D:\PlantXAI-Net\dataset\PlantVillage\balanced")

print("-" * 70)

for class_folder in sorted(BALANCED_DATASET.iterdir()):
    if class_folder.is_dir():
        total = len(list(class_folder.glob("*.*")))
        print(f"{class_folder.name:40} {total}")

print("-" * 70)