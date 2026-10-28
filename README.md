# PlantXAI-Net

Explainable AI (XAI) framework for Plant Disease Detection and Recommendation system.

## Project Structure

```text
PlantXAI-Net/
├── dataset/
│   ├── PlantVillage/
│   └── PlantDoc/
├── preprocessing/
├── models/
│   └── training_pipeline.py  # EfficientNet-B0 Transfer Learning Pipeline
├── segmentation/
├── xai/                      # SHAP and Captum integration
├── recommendation/
├── severity/                 # Disease severity estimation
├── results/
│   ├── checkpoints/
│   ├── logs/
│   ├── metrics/
│   ├── plots/
│   └── reports/
├── research_paper/
├── docs/
│   ├── PROJECT_OVERVIEW.md
│   └── DETAILED_REPORT.md
├── main.py
├── requirements.txt
└── README.md
```

## Features
- **Deep Learning Model:** ImageNet Pretrained EfficientNet-B0.
- **Explainable AI (XAI):** Integrated with SHAP and Captum for model interpretability.
- **Advanced Training:** Mixed Precision Training (AMP), Early Stopping, and Learning Rate Scheduling.
- **Comprehensive Evaluation:** Accuracy, Precision, Recall, F1-Score, Confusion Matrix, and Classification Reports.

## Getting Started

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run main application:**
   ```bash
   python main.py
   ```

## Documentation
- [Project Overview](docs/PROJECT_OVERVIEW.md)
- [Detailed Report](docs/DETAILED_REPORT.md)
