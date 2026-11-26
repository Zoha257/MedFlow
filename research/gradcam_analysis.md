# Grad-CAM analysis

This report was generated from examples in the PathMNIST test split using `resnet18.pth`.

## How the examples were selected

The first 10 correct and the first 10 incorrect predictions in test-split order were taken. This is **not** a random or class-balanced sample, so the share of correct examples here says nothing about model accuracy. Use `results/resnet18/results.json` for accuracy.

## What the selected examples contain

- Examples: 20 (10 correct, 10 incorrect)
- True classes among correct examples: colorectal adenocarcinoma epithelium (4), mucus (4), adipose (1), normal colon mucosa (1)
- True classes among incorrect examples: adipose (6), debris (2), colorectal adenocarcinoma epithelium (1), smooth muscle (1)
- Predicted classes among incorrect examples: smooth muscle (8), colorectal adenocarcinoma epithelium (1), normal colon mucosa (1)
- Target layer: `ResNet18.layer4[-1]` (4 x 4 feature map upsampled to 28 x 28, so heatmaps are coarse)

## Interpretation

Grad-CAM heatmaps show regions that contributed most to the selected class score. They are model-attribution visualizations, not proof that the model used clinically meaningful tissue structures. Conclusions apply only to the displayed samples.

## Correct predictions

- `correct_01.png`: true **colorectal adenocarcinoma epithelium**, predicted **colorectal adenocarcinoma epithelium**
- `correct_02.png`: true **mucus**, predicted **mucus**
- `correct_03.png`: true **mucus**, predicted **mucus**
- `correct_04.png`: true **colorectal adenocarcinoma epithelium**, predicted **colorectal adenocarcinoma epithelium**
- `correct_05.png`: true **mucus**, predicted **mucus**
- `correct_06.png`: true **mucus**, predicted **mucus**
- `correct_07.png`: true **colorectal adenocarcinoma epithelium**, predicted **colorectal adenocarcinoma epithelium**
- `correct_08.png`: true **adipose**, predicted **adipose**
- `correct_09.png`: true **normal colon mucosa**, predicted **normal colon mucosa**
- `correct_10.png`: true **colorectal adenocarcinoma epithelium**, predicted **colorectal adenocarcinoma epithelium**

## Incorrect predictions

- `wrong_01.png`: true **colorectal adenocarcinoma epithelium**, predicted **normal colon mucosa**
- `wrong_02.png`: true **adipose**, predicted **smooth muscle**
- `wrong_03.png`: true **smooth muscle**, predicted **colorectal adenocarcinoma epithelium**
- `wrong_04.png`: true **adipose**, predicted **smooth muscle**
- `wrong_05.png`: true **debris**, predicted **smooth muscle**
- `wrong_06.png`: true **adipose**, predicted **smooth muscle**
- `wrong_07.png`: true **debris**, predicted **smooth muscle**
- `wrong_08.png`: true **adipose**, predicted **smooth muscle**
- `wrong_09.png`: true **adipose**, predicted **smooth muscle**
- `wrong_10.png`: true **adipose**, predicted **smooth muscle**

## Manual review (fill in after looking at the figures)

Heatmap relevance cannot be computed automatically. For each category below, view the figures in `paper/figures/` and record what you actually see.

- Correct prediction + sensible heatmap: _(file names and what the heatmap covers)_
- Correct prediction + suspicious heatmap: _(file names; e.g. background, borders, artefacts)_
- Wrong prediction + heatmap: _(file names; what the model attended to, which class it confused)_
- Patterns across classes: _(which classes are confused, and whether the highlighted regions explain it)_
