# PlantXAI-Net Detailed Technical Report

## 1. Abstract
This report details the technical implementation and methodology of **PlantXAI-Net**, a comprehensive framework for plant disease detection leveraging Deep Learning and Explainable Artificial Intelligence (XAI). The system utilizes an EfficientNet-B0 architecture trained on the PlantVillage dataset and incorporates XAI techniques to ensure transparent and trustworthy predictions.

## 2. Dataset and Preprocessing
The primary dataset used for model training and validation is the **PlantVillage** dataset, which consists of thousands of healthy and diseased plant leaf images categorized by species and disease type.

### 2.1 Data Augmentation
To enhance the model's generalization capabilities and prevent overfitting, the following augmentations are applied during training:
- **Resizing:** All images are resized to $224 \times 224$ pixels.
- **Random Horizontal Flip:** Applied with a probability of 50%.
- **Random Rotation:** Images are randomly rotated by up to 15 degrees.
- **Color Jitter:** Random adjustments to brightness, contrast, saturation, and hue.
- **Normalization:** Images are normalized using standard ImageNet mean (`[0.485, 0.456, 0.406]`) and standard deviation (`[0.229, 0.224, 0.225]`).

## 3. Model Architecture and Training Pipeline
The core classification model is **EfficientNet-B0**, selected for its optimal balance between accuracy and computational efficiency. The model is initialized with ImageNet pre-trained weights to leverage Transfer Learning.

### 3.1 Training Configurations
- **Batch Size:** 32
- **Optimizer:** Adam/SGD variants with Weight Decay ($1e-4$)
- **Learning Rate:** Initialized at $0.001$, dynamically adjusted using a Learning Rate Scheduler.
- **Mixed Precision (AMP):** Utilized on CUDA-enabled devices to accelerate training and reduce memory footprint.

### 3.2 Regularization and Checkpointing
- **Early Stopping:** Training halts if the validation loss does not improve for 7 consecutive epochs (patience = 7).
- **Model Checkpointing:** The pipeline automatically saves the "best" model based on validation performance, ensuring the optimal weights are retained for inference.

## 4. Evaluation Metrics
The framework employs a rigorous evaluation methodology to assess model performance across all classes. The metrics recorded include:
- **Overall Accuracy:** The percentage of correctly classified images.
- **Precision, Recall, and F1-Score:** Calculated for each class to handle potential dataset imbalances.
- **Confusion Matrix:** Generated to visually identify specific inter-class misclassifications (e.g., distinguishing between different types of blights).
- **Classification Report:** Detailed statistical breakdown saved in the `results/reports/` directory.

## 5. Explainable AI (XAI) Integration
To interpret the model's predictions, PlantXAI-Net integrates **SHAP (SHapley Additive exPlanations)** and **Captum** (PyTorch's model interpretability library).

### 5.1 Methodology
When an image is passed through the model for inference, the XAI module computes the attribution scores for each pixel. These scores indicate how much each pixel positively or negatively contributed to the final class prediction.
- **Heatmaps:** The attribution scores are overlaid on the original image as heatmaps, clearly highlighting the diseased lesions, spots, or discoloration that the model focused on.
- **Trust and Validation:** This step is critical for agronomists to verify that the model is making decisions based on actual biological symptoms and not spurious background correlations.

## 6. Future Scope
Future iterations of PlantXAI-Net will focus on:
- Enhancing the **Severity Estimation** module using precise image segmentation algorithms (e.g., Mask R-CNN or U-Net) to quantify the percentage of affected leaf area.
- Expanding the dataset to include field-captured images (e.g., **PlantDoc** dataset) with complex backgrounds.
- Deploying the inference engine to edge devices (mobile applications) for real-time field diagnosis.
