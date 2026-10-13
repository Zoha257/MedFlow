from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from medmnist import PathMNIST

ROOT_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = ROOT_DIR / "paper" / "figures"
DATASET_SUMMARY = ROOT_DIR / "data" / "processed" / "dataset_summary.json"
LABELS = [
    "adipose",
    "background",
    "debris",
    "lymphocytes",
    "mucus",
    "smooth muscle",
    "normal colon mucosa",
    "cancer-associated stroma",
    "colorectal adenocarcinoma epithelium",
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def generate_dataset_distribution() -> dict[str, list[int]]:
    counts: dict[str, list[int]] = {}
    for split in ("train", "val", "test"):
        dataset = PathMNIST(split=split, download=False)
        labels = np.asarray(dataset.labels).reshape(-1)
        counts[split] = np.bincount(labels, minlength=len(LABELS)).tolist()

    positions = np.arange(len(LABELS))
    width = 0.25
    figure, axis = plt.subplots(figsize=(13, 6))
    for index, split in enumerate(("train", "val", "test")):
        axis.bar(positions + (index - 1) * width, counts[split], width, label=split)
    axis.set(
        title="PathMNIST class distribution",
        xlabel="Class",
        ylabel="Images",
        xticks=positions,
        xticklabels=LABELS,
    )
    axis.tick_params(axis="x", rotation=35)
    axis.legend()
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "dataset_distribution.png", dpi=180)
    plt.close(figure)
    return counts


def generate_training_curves(comparison: dict[str, dict[str, Any]]) -> None:
    figure, axis = plt.subplots(figsize=(9, 5))
    for name, metrics in comparison.items():
        history = metrics.get("history", [])
        if not history:
            continue
        axis.plot(
            [entry["epoch"] for entry in history],
            [entry["val_accuracy"] for entry in history],
            marker="o",
            label=name,
        )
    axis.set(title="Validation accuracy by epoch", xlabel="Epoch", ylabel="Accuracy")
    axis.legend()
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "training_curves.png", dpi=180)
    plt.close(figure)


def generate_model_comparison(comparison: dict[str, dict[str, Any]]) -> None:
    names = list(comparison)
    accuracy = [comparison[name]["accuracy"] for name in names]
    f1 = [comparison[name]["f1_macro"] for name in names]
    positions = np.arange(len(names))
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(positions - 0.2, accuracy, 0.4, label="Accuracy")
    axis.bar(positions + 0.2, f1, 0.4, label="Macro F1")
    axis.set(
        title="Measured model comparison",
        ylabel="Score",
        xticks=positions,
        xticklabels=names,
    )
    axis.set_ylim(0, 1)
    axis.legend()
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "model_comparison.png", dpi=180)
    plt.close(figure)


def generate_architecture_diagram() -> None:
    figure, axis = plt.subplots(figsize=(12, 3))
    axis.axis("off")
    blocks = [
        ("PathMNIST\n28 x 28 x 3", 0.08),
        ("Validation +\nToTensor", 0.27),
        ("ResNet18\nlayer4[-1]", 0.48),
        ("Linear\n9 classes", 0.69),
        ("Prediction +\nGrad-CAM", 0.88),
    ]
    for index, (label, x) in enumerate(blocks):
        axis.text(
            x,
            0.5,
            label,
            ha="center",
            va="center",
            fontsize=11,
            bbox={"boxstyle": "round,pad=0.7", "facecolor": "#dceeff", "edgecolor": "#31688e"},
        )
        if index < len(blocks) - 1:
            axis.annotate("", xy=(blocks[index + 1][1] - 0.07, 0.5), xytext=(x + 0.07, 0.5), arrowprops={"arrowstyle": "->"})
    axis.set_title("MedFlow-XAI inference and explanation architecture")
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "architecture.png", dpi=180)
    plt.close(figure)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    comparison = load_json(RESULTS_DIR / "model_comparison.json")
    dataset_counts = generate_dataset_distribution()
    generate_training_curves(comparison)
    generate_model_comparison(comparison)
    generate_architecture_diagram()
    (FIGURES_DIR / "dataset_distribution.json").write_text(
        json.dumps({"labels": LABELS, "counts": dataset_counts}, indent=2),
        encoding="utf-8",
    )
    if DATASET_SUMMARY.exists():
        print(f"Used dataset summary: {DATASET_SUMMARY}")
    print(f"Generated research figures in {FIGURES_DIR}")


if __name__ == "__main__":
    main()
