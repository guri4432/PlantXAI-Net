import os
import pandas as pd

# Change this path according to your system
dataset_path = r"D:\PlantXAI-Net\dataset\PlantVillage\color"

data = []

for class_name in sorted(os.listdir(dataset_path)):
    class_path = os.path.join(dataset_path, class_name)

    if os.path.isdir(class_path):
        image_count = len([
            file for file in os.listdir(class_path)
            if file.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

        data.append({
            "Class": class_name,
            "Images": image_count
        })

df = pd.DataFrame(data)

print(df)

print("\nTotal Classes :", len(df))
print("Total Images :", df["Images"].sum())

# Save CSV
df.to_csv("results/class_distribution.csv", index=False)

print("\nClass distribution saved successfully.")