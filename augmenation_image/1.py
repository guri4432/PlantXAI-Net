import cv2
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Original image path
# ============================================================

IMAGE_PATH = r"D:\PlantXAI-Net\dataset\PlantVillage\split\train\Apple___Apple_scab\0a769a71-052a-4f19-a4d8-b0f0cb75541c___FREC_Scab 3165.JPG"

# ============================================================
# Load image
# ============================================================

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(
        f"Image not found: {IMAGE_PATH}"
    )

image = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)

# ============================================================
# 1. Original
# ============================================================

original = image.copy()

# ============================================================
# 2. Horizontal Flip
# ============================================================

flip = cv2.flip(
    image,
    1
)

# ============================================================
# 3. Rotation
# ============================================================

height, width = image.shape[:2]

center = (
    width // 2,
    height // 2
)

rotation_matrix = cv2.getRotationMatrix2D(
    center,
    25,
    1.0
)

rotation = cv2.warpAffine(
    image,
    rotation_matrix,
    (width, height)
)

# ============================================================
# 4. Brightness / Contrast
# ============================================================

brightness_contrast = cv2.convertScaleAbs(
    image,
    alpha=1.20,
    beta=25
)

# ============================================================
# 5. Geometric Transformation
# ============================================================

src_points = np.float32([
    [0, 0],
    [width - 1, 0],
    [0, height - 1]
])

dst_points = np.float32([
    [20, 20],
    [width - 40, 10],
    [30, height - 30]
])

affine_matrix = cv2.getAffineTransform(
    src_points,
    dst_points
)

geometric = cv2.warpAffine(
    image,
    affine_matrix,
    (width, height)
)

# ============================================================
# 6. Gamma Adjustment
# ============================================================

gamma = 1.4

normalized = image / 255.0

gamma_image = np.power(
    normalized,
    gamma
)

gamma_image = np.uint8(
    gamma_image * 255
)

# ============================================================
# Create Figure
# ============================================================

fig, axes = plt.subplots(
    2,
    3,
    figsize=(12, 8)
)

images = [
    original,
    flip,
    rotation,
    brightness_contrast,
    geometric,
    gamma_image
]

titles = [
    "(a) Original",
    "(b) Horizontal Flip",
    "(c) Rotation",
    "(d) Brightness/Contrast",
    "(e) Geometric Transformation",
    "(f) Gamma Adjustment"
]

for ax, img, title in zip(
    axes.ravel(),
    images,
    titles
):

    ax.imshow(img)

    ax.set_title(
        title,
        fontsize=11
    )

    ax.axis("off")

plt.tight_layout()

# ============================================================
# Save Figure
# ============================================================

OUTPUT_PATH = r"D:\PlantXAI-Net\results\augmentation_examples.png"

plt.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print()
print("Augmentation figure saved:")
print(OUTPUT_PATH)