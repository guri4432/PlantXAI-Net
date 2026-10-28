import os
from PIL import Image

dataset_path = r"D:\PlantXAI-Net\dataset\PlantVillage\color"

total_images = 0
corrupted_images = []

for class_name in sorted(os.listdir(dataset_path)):
    class_path = os.path.join(dataset_path, class_name)

    if not os.path.isdir(class_path):
        continue

    for image_name in os.listdir(class_path):
        image_path = os.path.join(class_path, image_name)

        try:
            with Image.open(image_path) as img:
                img.verify()
            total_images += 1
        except Exception:
            corrupted_images.append(image_path)

print("=" * 60)
print(f"Total Images Checked : {total_images}")
print(f"Corrupted Images     : {len(corrupted_images)}")
print("=" * 60)

if corrupted_images:
    print("\nCorrupted image paths:")
    for path in corrupted_images:
        print(path)
else:
    print("\n✅ Dataset is clean.")