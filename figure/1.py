import matplotlib.pyplot as plt

# ============================================================
# Training History
# ============================================================

epochs = [
    1, 2, 3, 4,
    5, 6, 7, 8
]

train_loss = [
    0.6889,
    0.3618,
    0.3254,
    0.3114,
    0.2959,
    0.2959,
    0.2903,
    0.2905
]

val_loss = [
    0.2178,
    0.1787,
    0.1554,
    0.1336,
    0.1346,
    0.1293,
    0.1325,
    0.1260
]

train_accuracy = [
    84.52,
    89.61,
    90.23,
    90.59,
    91.01,
    91.06,
    91.01,
    91.07
]

val_accuracy = [
    93.66,
    94.77,
    95.36,
    95.89,
    95.60,
    96.08,
    95.56,
    95.71
]


# ============================================================
# Figure 1 — Training and Validation Loss
# ============================================================

plt.figure(figsize=(7, 5))

plt.plot(
    epochs,
    train_loss,
    marker="o",
    label="Training Loss"
)

plt.plot(
    epochs,
    val_loss,
    marker="s",
    label="Validation Loss"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.title(
    "Training and Validation Loss"
)

plt.xticks(
    epochs
)

plt.grid(
    True,
    alpha=0.3
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "training_validation_loss.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# Figure 2 — Training and Validation Accuracy
# ============================================================

plt.figure(figsize=(7, 5))

plt.plot(
    epochs,
    train_accuracy,
    marker="o",
    label="Training Accuracy"
)

plt.plot(
    epochs,
    val_accuracy,
    marker="s",
    label="Validation Accuracy"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Accuracy (%)"
)

plt.title(
    "Training and Validation Accuracy"
)

plt.xticks(
    epochs
)

plt.grid(
    True,
    alpha=0.3
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "training_validation_accuracy.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# Final Information
# ============================================================

print()
print("=" * 70)
print("TRAINING CURVES GENERATED")
print("=" * 70)

print(
    "Training/Validation Loss : training_validation_loss.png"
)

print(
    "Training/Validation Accuracy : "
    "training_validation_accuracy.png"
)

print(
    "Best Validation Accuracy : "
    f"{max(val_accuracy):.2f}% "
    f"(Epoch {epochs[val_accuracy.index(max(val_accuracy))]})"
)

print(
    "Lowest Validation Loss : "
    f"{min(val_loss):.4f} "
    f"(Epoch {epochs[val_loss.index(min(val_loss))]})"
)

print("=" * 70)