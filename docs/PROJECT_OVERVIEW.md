# PlantXAI-Net Project Overview

## Introduction
**PlantXAI-Net** is an advanced Explainable AI (XAI) framework designed for robust plant disease detection, severity estimation, and treatment recommendation. By leveraging state-of-the-art Deep Learning techniques combined with XAI, this project aims to provide agronomists and farmers with not only accurate disease classifications but also transparent, interpretable visual explanations of the model's decision-making process.

## Core Objectives
1. **Accurate Disease Detection:** Utilize Transfer Learning with state-of-the-art Convolutional Neural Networks (EfficientNet-B0) to classify various plant leaf diseases with high accuracy.
2. **Model Interpretability (XAI):** Demystify the "black-box" nature of Deep Learning models using SHAP (SHapley Additive exPlanations) and Captum, highlighting the exact regions of the leaf that contribute most to the predicted disease.
3. **Severity Estimation & Segmentation:** Go beyond simple classification by segmenting the diseased areas to estimate the severity of the infection.
4. **Treatment Recommendation:** Provide actionable agricultural recommendations based on the diagnosed disease and its estimated severity.

## Key Components

### 1. Data Pipeline & Preprocessing
The framework relies on large-scale agricultural datasets like **PlantVillage** and **PlantDoc**. The data pipeline handles image resizing, normalization (ImageNet standards), and rigorous data augmentation (random flips, rotations, color jittering) to ensure model robustness.

### 2. Deep Learning Pipeline
At the core of the classification engine is **EfficientNet-B0**, a highly efficient and accurate architecture. The training pipeline features:
- Mixed Precision Training (AMP) for faster convergence.
- Early Stopping to prevent overfitting.
- Learning Rate Schedulers.
- Comprehensive checkpointing and logging.

### 3. Explainable AI (XAI) Module
This module applies post-hoc interpretability techniques (SHAP and Captum) to generate heatmaps and attribution masks over the input images. This ensures that the model focuses on actual disease symptoms (e.g., spots, lesions, discoloration) rather than background artifacts.

### 4. Recommendation & Severity Modules
Once a disease is detected, the severity module calculates the extent of the damage. Based on the specific disease and its severity, the recommendation system queries a knowledge base to suggest chemical or biological treatments, preventive measures, and optimal cultural practices.

## Conclusion
PlantXAI-Net aims to bridge the gap between complex AI research and practical, trustworthy agricultural applications. By providing explainable predictions, it fosters trust among end-users and paves the way for smarter, AI-driven farming practices.
