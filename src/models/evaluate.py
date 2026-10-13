from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader

from .cnn import BaselineCNN


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int = 9,
) -> Tuple[Dict[str, float], np.ndarray]:
    model.eval()
    predictions: list[int] = []
    targets: list[int] = []

    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device))
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            targets.extend(labels.view(-1).long().tolist())

    matrix = confusion_matrix(targets, predictions, labels=list(range(num_classes)))
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets,
        predictions,
        labels=list(range(num_classes)),
        average="macro",
        zero_division=0,
    )
    metrics = {
        "accuracy": float(accuracy_score(targets, predictions)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
    }
    return metrics, matrix


def save_evaluation_artifacts(
    metrics: Dict[str, float],
    matrix: np.ndarray,
    output_dir: Path,
    labels: Iterable[str] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    class_labels = list(labels) if labels is not None else [str(i) for i in range(matrix.shape[0])]
    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        xlabel="Predicted label",
        ylabel="True label",
        title="PathMNIST confusion matrix",
        xticks=np.arange(len(class_labels)),
        yticks=np.arange(len(class_labels)),
        xticklabels=class_labels,
        yticklabels=class_labels,
    )
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                int(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )
    figure.tight_layout()
    figure.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close(figure)


def load_baseline_model(checkpoint: Path, device: torch.device) -> BaselineCNN:
    model = BaselineCNN()
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model.to(device)
