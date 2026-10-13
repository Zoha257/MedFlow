# MedFlow-XAI: An Explainable and Data-Centric Pipeline for Medical Image Classification

## Abstract

MedFlow-XAI is a reproducible research pipeline for classifying small
histopathology image patches from the official PathMNIST dataset. The project
connects dataset acquisition and validation, preprocessing, PostgreSQL
metadata, supervised learning, MLflow experiment tracking, Grad-CAM
visualization, and a FastAPI inference service. The Day 2 baseline CNN was
trained for one epoch on the full training split and achieved a measured test
accuracy of 0.6753 and macro F1 of 0.5923. The available ResNet18 records in
this worktree are bounded smoke runs using 32 samples; they achieved 0.0625
test accuracy and 0.0131 macro F1 and therefore are not treated as a valid
full-dataset comparison. Grad-CAM examples were generated from the available
checkpoint, but the examples expose class-collapse behavior: the selected
correct examples are background images and the selected errors are predicted
as background. This result is a useful failure record rather than evidence of
clinical utility. The system is explicitly educational and research-oriented,
not a diagnostic tool. The report documents the measured behavior, the
deployment interfaces, the limitations of the current runs, and the next
experiments required for a defensible medical-imaging evaluation.

## 1. Introduction

Medical image classification can support research workflows by making
repeatable measurements over large collections of images. Histopathology is a
particularly relevant setting because images contain visual patterns at
multiple scales and because model errors may be difficult to inspect from a
single accuracy value. A trustworthy workflow therefore needs more than a
classifier: it needs dataset provenance, validation, reproducible training,
error analysis, explainability, and a clear boundary between research output
and clinical decision-making.

The research question for this project is: **Can a reproducible,
data-centric pipeline combine PathMNIST classification, experiment tracking,
visual explanations, and an API while making model limitations visible?**
The answer is evaluated as an engineering and research workflow, not as a
claim that the resulting model is clinically deployable.

## 2. Literature and dataset context

PathMNIST is part of MedMNIST, a standardized collection designed to make
medical image classification experiments accessible and reproducible. The
dataset used here contains 28 x 28 RGB patches and nine tissue classes. The
official source and dataset documentation are cited in the project README and
the references below.

Explainability is addressed with Grad-CAM. Grad-CAM uses gradients flowing
into a convolutional layer to weight activation maps and produce a spatial
importance map for a selected class. These maps are diagnostic aids for
inspecting model behavior; they do not establish causal reasoning, clinical
validity, or biological relevance.

## 3. Methodology

### 3.1 Dataset and preprocessing

The official MedMNIST `PathMNIST` loader supplies train, validation, and test
splits. The recorded dataset summary contains 89,996 training images, 10,004
validation images, and 7,180 test images. Images are converted to tensors and
kept at the dataset's 28 x 28 RGB resolution. The Day 3 augmentation branch
applies random horizontal and vertical flips and a 15-degree rotation only to
training images.

Dataset metadata is stored through the PostgreSQL database layer when the
configured local server is available. The code also records the image shape
and split sizes in `data/processed/dataset_summary.json`.

### 3.2 Models and evaluation

The baseline is a compact CNN with two convolutional blocks and a fully
connected classifier. The ResNet18 model adapts the first convolution and
removes the ImageNet max-pooling operation so it can process 28 x 28 images.
The final fully connected layer outputs nine classes.

Evaluation reports accuracy, macro precision, macro recall, macro F1, and a
nine-class confusion matrix. Training records loss and validation metrics by
epoch. MLflow stores parameters and metrics for Day 3 experiments, while
PostgreSQL stores experiment metadata when PostgreSQL is running.

### 3.3 Explainability

Grad-CAM targets `ResNet18.layer4[-1]`. For each selected prediction, the
implementation backpropagates the predicted-class score, averages gradients
over spatial dimensions, weights the feature maps, applies ReLU, and
normalizes the result to the input resolution. The generated overlay is
saved as a PNG and the manifest records the true and predicted classes.

### 3.4 Deployment

The FastAPI service exposes `/health`, `/predict`, and `/explain`. Uploaded
images are validated, limited to 10 MB, converted to RGB, resized to 28 x 28,
and passed to the same ResNet18 preprocessing path. `/explain` returns a
base64-encoded PNG Grad-CAM overlay. Docker files package the API and the
ResNet18 checkpoint for local deployment.

## 4. Results

### 4.1 Dataset

The generated `paper/figures/dataset_distribution.png` is based on labels
read from the local PathMNIST splits. The exact counts are stored alongside
the figure in `dataset_distribution.json`. No synthetic labels or manually
entered counts are used.

### 4.2 Measured model results

| Model | Accuracy | Macro F1 | Training time | Run scope |
|---|---:|---:|---:|---|
| BaselineCNN | 0.6753 | 0.5923 | 776.61 s | Full dataset, 1 epoch |
| ResNet18 | 0.0625 | 0.0131 | 21.96 s | Smoke run, 32 samples |
| ResNet18 + augmentation | 0.0625 | 0.0131 | 34.67 s | Smoke run, 32 samples |

The ResNet values must not be interpreted as a fair comparison with the
baseline. A full-dataset ResNet run is required before making a model-ranking
claim. The training curves and comparison figure visualize the values recorded
in `results/model_comparison.json`.

### 4.3 Explainability outputs

The current Grad-CAM generation produced ten correct and ten incorrect
examples. All selected correct examples were from the background class. The
selected incorrect examples included mucus, colorectal adenocarcinoma
epithelium, adipose, and normal colon mucosa, all predicted as background.
This pattern is consistent with class collapse in the available smoke-run
checkpoint. The figures are retained as an auditable failure case, not as a
representative final explanation set.

## 5. Error analysis

The dominant observed error is prediction of non-background examples as
background. The most likely engineering explanation is insufficient training
signal from the bounded 32-sample smoke run, not a conclusion about the
dataset or tissue morphology. The available evidence does not justify
claiming that the heatmaps identify relevant tissue regions. The next
diagnostic steps are a full-data ResNet run, per-class support and recall,
confusion-matrix inspection, and review of representative explanations after
the model has learned all classes.

## 6. Limitations

The Streamlit interface is implemented, but no browser-based user study or
production deployment was performed. PostgreSQL was unavailable during the
local Day 3 smoke run, so experiment inserts were skipped even though the database integration code is present. The ResNet
records are smoke tests rather than a full training study, and only one epoch
was used for the baseline. The selected Grad-CAM examples are not a
statistically representative explanation sample. No clinical validation,
external test set, patient-level split analysis, calibration study, or
prospective evaluation was performed.

## 7. Conclusion

The project demonstrates an end-to-end structure for a medical-AI research
pipeline: validated data, measurable training, database and MLflow hooks,
visual attribution, and an HTTP deployment boundary. The most important
finding is methodological: reproducibility and honest failure reporting expose
that the current ResNet checkpoint is not ready for comparative conclusions.
The baseline result is measurable, while the ResNet and Grad-CAM artifacts
identify the exact next experiment rather than hiding a class-collapse
failure. A defensible continuation is to complete Day 6, run full-data
ResNet experiments, verify PostgreSQL persistence, and repeat error analysis
before drawing scientific conclusions. The Day 6 frontend is now available
for local demonstration, but its output should be interpreted with the same
research-only limitations as the API.

## References

1. MedMNIST project and official dataset documentation:
   <https://github.com/MedMNIST/MedMNIST>
2. Selvaraju et al., “Grad-CAM: Visual Explanations from Deep Networks via
   Gradient-based Localization,” <https://arxiv.org/abs/1610.02391>
3. DEAI course content: <https://deai.ulb.be/home/master-programme/course-content/>
4. EDISS first-year curriculum: <https://www.master-ediss.eu/first-year/>
5. MAIA course information: <https://maiamaster.udg.edu/courses/>
