# Grad-CAM analysis

This report was generated from selected examples in the PathMNIST test split using `resnet18.pth`.

## Evaluation summary

- Evaluated images: 20
- Correct predictions: 10
- Incorrect predictions: 10
- Accuracy among selected examples: 0.5000
- Target layer: `ResNet18.layer4[-1]`

## Interpretation

The heatmaps show regions that contributed most to the selected class score. They are model-attribution visualizations, not proof that the model used clinically meaningful tissue structures.
Review the saved correct and incorrect examples below for tissue relevance and failure analysis. Conclusions should be limited to the displayed samples and should not be generalized to clinical use.
In this run, all selected correct examples were from the background class, while the selected errors were non-background images predicted as background. This indicates a class-collapse failure in the current checkpoint; it should be re-trained on the full dataset before using these figures as representative model behavior.

## Correct predictions

- `correct_01.png`: true **background**, predicted **background**
- `correct_02.png`: true **background**, predicted **background**
- `correct_03.png`: true **background**, predicted **background**
- `correct_04.png`: true **background**, predicted **background**
- `correct_05.png`: true **background**, predicted **background**
- `correct_06.png`: true **background**, predicted **background**
- `correct_07.png`: true **background**, predicted **background**
- `correct_08.png`: true **background**, predicted **background**
- `correct_09.png`: true **background**, predicted **background**
- `correct_10.png`: true **background**, predicted **background**

## Incorrect predictions

- `wrong_01.png`: true **colorectal adenocarcinoma epithelium**, predicted **background**
- `wrong_02.png`: true **mucus**, predicted **background**
- `wrong_03.png`: true **mucus**, predicted **background**
- `wrong_04.png`: true **colorectal adenocarcinoma epithelium**, predicted **background**
- `wrong_05.png`: true **mucus**, predicted **background**
- `wrong_06.png`: true **mucus**, predicted **background**
- `wrong_07.png`: true **colorectal adenocarcinoma epithelium**, predicted **background**
- `wrong_08.png`: true **adipose**, predicted **background**
- `wrong_09.png`: true **normal colon mucosa**, predicted **background**
- `wrong_10.png`: true **colorectal adenocarcinoma epithelium**, predicted **background**
