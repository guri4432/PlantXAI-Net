# PlantXAI-Net 🌿
### An Explainable AI Framework for Plant Disease Classification, Severity Estimation, and Action Guidance

PlantXAI-Net is an explainable artificial intelligence (XAI) framework designed to support plant disease analysis through image classification, visual explanations, Grad-CAM-derived affected-area estimation, and confidence-aware action guidance.

The framework integrates deep learning and interpretable visual analysis to make model predictions easier to inspect. It is intended as a research prototype for plant disease decision support, not as a replacement for expert agronomic diagnosis.

## Key Features

- **Plant disease classification:** EfficientNet-B0 as the primary classification model.
- **Baseline comparison:** ResNet-50 evaluated under the project's reported experimental setup.
- **Explainable AI:** Grad-CAM heatmaps to visualize image regions contributing to model predictions.
- **Severity estimation:** Affected-area proxy derived from thresholded Grad-CAM heatmaps and an estimated leaf mask.
- **Confidence-aware action guidance:** Routes predictions to guidance categories or manual verification.
- **Evaluation and visualization:** Training histories, classification metrics, confusion matrices, and XAI/severity result files.

## Framework Overview

The pipeline follows these stages:

1. Input plant-leaf image.
2. Image preprocessing and model inference.
3. Disease-class prediction and confidence estimation.
4. Grad-CAM heatmap generation.
5. Estimated affected-area calculation and severity categorization.
6. Confidence-aware action guidance or manual verification.

**Important:** The affected-area estimate is a Grad-CAM-derived proxy. It is not equivalent to a validated lesion segmentation mask or expert-confirmed disease severity.

## Dataset

The primary experiments use the PlantVillage dataset with 38 healthy/disease classes.

| Split | Images | Percentage |
|---|---:|---:|
| Training | 65,827 | 85.82% |
| Validation | 5,428 | 7.08% |
| Test | 5,448 | 7.10% |
| **Total** | **76,703** | **100%** |

- Input resolution: 224 × 224 pixels.
- Input format: RGB images.
- Training data: balanced training set with augmentation.
- Validation and test data: used for model selection and evaluation, respectively.

The dataset itself is not included in this repository. Obtain it from an authorized source and follow the dataset's applicable terms and conditions.

## Models and Experimental Setup

### EfficientNet-B0 — Primary Model

The primary classifier uses an ImageNet-pretrained EfficientNet-B0 with a frozen feature-extraction backbone and a trainable task-specific classification head.

| Parameter | Configuration |
|---|---|
| Architecture | EfficientNet-B0 |
| Initialization | ImageNet pretrained |
| Classes | 38 |
| Image size | 224 × 224 |
| Batch size | 32 |
| Epochs completed | 8 |
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Weight decay | 0.0001 |
| Loss function | Cross-entropy |
| Learning-rate scheduler | ReduceLROnPlateau |
| Training device | CPU |
| Configured random seed | 42 |

### ResNet-50 — Baseline

ResNet-50 was evaluated as a baseline using ImageNet-pretrained weights and a frozen backbone with a trainable classification head.

| Parameter | Value |
|---|---:|
| Total parameters | 23,585,894 |
| Trainable parameters | 77,862 |
| Frozen parameters | 23,508,032 |
| Best recorded validation accuracy | 97.24% |
| Test accuracy | 97.17% |

The reported test accuracy of ResNet-50 is higher than that of EfficientNet-B0 in these experiments. EfficientNet-B0 remains the primary model in the integrated framework; this selection should not be interpreted as an accuracy advantage over ResNet-50.

## Experimental Results

### Classification Performance

| Metric | EfficientNet-B0 | ResNet-50 |
|---|---:|---:|
| Test images | 5,448 | 5,448 |
| Correct predictions | 5,221 | — |
| Incorrect predictions | 227 | — |
| Test accuracy | 95.83% | 97.17% |
| Test loss | 0.1290 | 0.0964 |
| Weighted precision | Not recorded here | 97.26% |
| Weighted recall | Not recorded here | 97.17% |
| Weighted F1-score | Not recorded here | 97.14% |

ResNet-50 achieved a 1.34 percentage-point higher test accuracy in the recorded comparison.

These results are from the PlantVillage test split. They do not establish equivalent performance on field images or other external datasets.

### EfficientNet-B0 Training Summary

| Epoch | Training Accuracy | Validation Accuracy | Validation Loss |
|---:|---:|---:|---:|
| 1 | 84.52% | 93.66% | 0.2178 |
| 2 | 89.61% | 94.77% | 0.1787 |
| 3 | 90.23% | 95.36% | 0.1554 |
| 4 | 90.59% | 95.89% | 0.1336 |
| 5 | 91.01% | 95.60% | 0.1346 |
| 6 | 91.06% | 96.08% | 0.1293 |
| 7 | 91.01% | 95.56% | 0.1325 |
| 8 | 91.07% | 95.71% | 0.1261 |

The highest recorded validation accuracy was 96.08% at epoch 6. The lowest recorded validation loss was 0.1261 at epoch 8.

### Explainability with Grad-CAM

Grad-CAM generates heatmaps highlighting image regions that contribute to the classifier's prediction.

Recorded full-test-set analysis:

| Metric | Result |
|---|---:|
| Images analyzed | 5,448 |
| Classifier accuracy | 95.83% |
| Average prediction confidence | 95.25% |
| High confidence (≥90%) | 4,713 (86.51%) |
| Medium confidence (70% to <90%) | 414 (7.60%) |
| Low confidence (<70%) | 321 (5.89%) |

These confidence values describe model predictions, not the correctness of the visual explanations. Grad-CAM was evaluated qualitatively; expert-annotated lesion masks and quantitative localization metrics were not available.

### Severity Estimation

The severity module intersects a thresholded Grad-CAM heatmap with an estimated leaf mask to calculate a proxy for the affected leaf area.

**Affected-area proxy:**

Affected Area (%) = (Affected Leaf Pixels / Total Estimated Leaf Pixels) × 100

Configured heatmap threshold: 0.50.

| Severity category | Operational rule | Images | Percentage |
|---|---|---:|---:|
| Mild | Below 20% | 801 | 14.70% |
| Moderate | 20% to below 50% | 2,797 | 51.34% |
| Severe | 50% or above | 1,850 | 33.96% |
| **Total** | | **5,448** | **100%** |

The 20% and 50% boundaries are author-defined operational thresholds, not agronomist-calibrated boundaries. These category counts describe the model pipeline's outputs, not severity-classification accuracy.

### Confidence-Aware Action Guidance

The recommendation module uses a confidence threshold of 0.70 to route outputs.

- Mild severity with sufficient confidence → Routine Monitoring.
- Moderate severity with sufficient confidence → Recommended Action.
- Severe severity with sufficient confidence → Immediate Attention.
- Confidence below 0.70 → Manual Verification.

| Routing outcome | Images | Percentage |
|---|---:|---:|
| Direct guidance | 5,112 | 93.83% |
| Manual verification | 336 | 6.17% |
| **Total** | **5,448** | **100%** |

The 93.83% figure represents direct-guidance routing coverage, **not recommendation accuracy**. Independent expert recommendation labels and treatment-outcome validation were not available.

## Repository Structure

```text
PlantXAI-Net/
├── augmentation_image/
├── docs/
├── figure/
├── models/
├── preprocessing/
├── recommendation/
├── research_paper/
├── results/
│   ├── baselines/
│   ├── metrics/
│   ├── plots/
│   ├── recommendations/
│   ├── reports/
│   ├── severity/
│   └── xai/
├── segmentation/
├── severity/
├── xai/
├── app.py
├── main.py
├── requirements.txt
└── README.md
```

The exact contents may evolve as the project is updated.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/guri4432/PlantXAI-Net.git
cd PlantXAI-Net
```

### 2. Create a virtual environment

Windows:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Use a compatible Python version and resolve any platform-specific dependency requirements before running the project.

### 4. Prepare the dataset and model weights

The dataset and model checkpoints are not included in the repository. Obtain the required dataset and compatible model weights separately, then place them at the paths expected by the project configuration and source code.

### 5. Run the project

Inspect the entry points and configuration before execution.

```bash
python main.py
```

If you want to launch the interface, inspect the application's requirements and then run:

```bash
python app.py
```

The exact execution workflow may depend on the available dataset, checkpoint files, and installed dependencies.

## Reproducibility

- The configured random seed is 42.
- The reported EfficientNet-B0 and ResNet-50 experiments each have one recorded run.
- Reported metrics should be interpreted as results from those runs, not as averages over multiple independent trials.
- Results may vary with dependency versions, hardware, data ordering, and checkpoint selection.
- The model checkpoint files are excluded from the repository; reproducibility requires obtaining compatible weights or retraining the models.

## Limitations

1. **Controlled dataset:** The primary evaluation uses PlantVillage images. Generalization to field conditions, varied lighting, backgrounds, and camera devices remains unestablished.
2. **Severity validation:** The severity thresholds are operational choices and have not been calibrated using expert severity annotations.
3. **Qualitative explainability:** Grad-CAM maps have not been validated against expert-annotated pixel-level lesion masks.
4. **Recommendation evaluation:** Routing coverage is reported, but recommendation accuracy and treatment effectiveness have not been independently validated.
5. **Baseline performance:** ResNet-50 achieved higher recorded test accuracy than EfficientNet-B0.
6. **Statistical reliability:** Multiple independent runs and uncertainty estimates are needed to establish robustness.
7. **External validation:** Evaluation on an independent field-image dataset remains future work.

## Future Work

- Evaluate on independent field-image datasets, including PlantDoc where suitable data and evaluation are available.
- Validate severity estimates against expert annotations and lesion masks.
- Measure Grad-CAM localization using suitable quantitative metrics.
- Evaluate action guidance against expert recommendations.
- Conduct repeated experiments with multiple random seeds.
- Benchmark inference latency, model size, and computational requirements.
- Expand the disease-specific guidance knowledge base and document its sources.

## Intended Use

PlantXAI-Net is a research prototype for plant disease image analysis and explainable decision support. Its outputs should be treated as advisory and verified by qualified agricultural experts before making consequential crop-management or treatment decisions.

## Citation

If you use this project in research, cite the associated paper once its final bibliographic details are available.

**Paper title:** *PlantXAI-Net: An Explainable AI Framework for Plant Disease Classification, Severity Estimation, and Action Recommendation*

Please add the final authors, venue, year, DOI, and publication link when available.

## License

No license has been specified in this repository documentation yet. Until a suitable license is added, do not assume that others have permission to reuse, modify, or redistribute the code.

## Acknowledgements

The project uses PlantVillage data and ImageNet-pretrained model architectures. Please cite the relevant dataset and model sources when publishing or redistributing derived work.
