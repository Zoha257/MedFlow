from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from medmnist import PathMNIST
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from .gradcam import GradCAM
from .resnet import create_resnet18

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = ROOT_DIR / "models" / "resnet18.pth"
DEFAULT_OUTPUT = ROOT_DIR / "paper" / "figures"
DEFAULT_ANALYSIS = ROOT_DIR / "research" / "gradcam_analysis.md"
CLASS_LABELS = {
    0: "adipose",
    1: "background",
    2: "debris",
    3: "lymphocytes",
    4: "mucus",
    5: "smooth muscle",
    6: "normal colon mucosa",
    7: "cancer-associated stroma",
    8: "colorectal adenocarcinoma epithelium",
}


def save_overlay(image: torch.Tensor, cam: torch.Tensor, output_path: Path, title: str) -> None:
    image_array = image.detach().cpu().permute(1, 2, 0).numpy().clip(0, 1)
    cam_array = cam.detach().cpu().numpy()
    figure, axis = plt.subplots(figsize=(4, 4))
    axis.imshow(image_array)
    axis.imshow(cam_array, cmap="jet", alpha=0.45, vmin=0, vmax=1)
    axis.set_title(title)
    axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT_DIR).as_posix()
    except ValueError:
        return path.name


def _count_by(records: list[dict[str, Any]], key: str) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    for record in records:
        counts[record[key]] = counts.get(record[key], 0) + 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))


def _format_counts(counts: list[tuple[str, int]]) -> str:
    return ", ".join(f"{label} ({count})" for label, count in counts) or "none"


def create_analysis(
    records: list[dict[str, Any]],
    checkpoint: Path,
    output_path: Path,
) -> None:
    """Write a Grad-CAM report whose statements are computed from the records.

    Nothing here is hardcoded about model behaviour. Whether a heatmap highlights
    *relevant tissue* is a human judgement, so that part is left as a template.
    """
    correct = [record for record in records if record["correct"]]
    incorrect = [record for record in records if not record["correct"]]
    predicted_classes = {record["predicted_label"] for record in records}
    lines = [
        "# Grad-CAM analysis",
        "",
        "This report was generated from examples in the PathMNIST test split using "
        f"`{checkpoint.name}`.",
        "",
        "## How the examples were selected",
        "",
        "The first 10 correct and the first 10 incorrect predictions in test-split "
        "order were taken. This is **not** a random or class-balanced sample, so the "
        "share of correct examples here says nothing about model accuracy. Use "
        "`results/resnet18/results.json` for accuracy.",
        "",
        "## What the selected examples contain",
        "",
        f"- Examples: {len(records)} ({len(correct)} correct, {len(incorrect)} incorrect)",
        f"- True classes among correct examples: {_format_counts(_count_by(correct, 'true_label'))}",
        f"- True classes among incorrect examples: {_format_counts(_count_by(incorrect, 'true_label'))}",
        f"- Predicted classes among incorrect examples: {_format_counts(_count_by(incorrect, 'predicted_label'))}",
        f"- Target layer: `ResNet18.layer4[-1]` (4 x 4 feature map upsampled to 28 x 28, so heatmaps are coarse)",
        "",
    ]
    if len(predicted_classes) == 1:
        lines.extend(
            [
                "> **Warning: class collapse.** Every selected example, correct or not, was "
                f"predicted as **{next(iter(predicted_classes))}**. The checkpoint behaves like a "
                "constant classifier, so these heatmaps are not meaningful explanations. "
                "Retrain before drawing any conclusion.",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation",
            "",
            "Grad-CAM heatmaps show regions that contributed most to the selected class "
            "score. They are model-attribution visualizations, not proof that the model "
            "used clinically meaningful tissue structures. Conclusions apply only to the "
            "displayed samples.",
            "",
            "## Correct predictions",
            "",
        ]
    )
    lines.extend(
        f"- `{record['file']}`: true **{record['true_label']}**, "
        f"predicted **{record['predicted_label']}**"
        for record in correct[:10]
    )
    lines.extend(["", "## Incorrect predictions", ""])
    lines.extend(
        f"- `{record['file']}`: true **{record['true_label']}**, "
        f"predicted **{record['predicted_label']}**"
        for record in incorrect[:10]
    )
    lines.extend(
        [
            "",
            "## Manual review (fill in after looking at the figures)",
            "",
            "Heatmap relevance cannot be computed automatically. For each category below, "
            "view the figures in `paper/figures/` and record what you actually see.",
            "",
            "- Correct prediction + sensible heatmap: _(file names and what the heatmap covers)_",
            "- Correct prediction + suspicious heatmap: _(file names; e.g. background, borders, artefacts)_",
            "- Wrong prediction + heatmap: _(file names; what the model attended to, which class it confused)_",
            "- Patterns across classes: _(which classes are confused, and whether the highlighted regions explain it)_",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_gradcam(
    checkpoint: Path = DEFAULT_CHECKPOINT,
    output_dir: Path = DEFAULT_OUTPUT,
    analysis_path: Path = DEFAULT_ANALYSIS,
    batch_size: int = 64,
    max_samples: int | None = None,
) -> dict[str, Any]:
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"ResNet18 checkpoint not found at {checkpoint}. "
            "Run the full Day 3 ResNet18 experiment first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = PathMNIST(
        split="test",
        transform=transforms.ToTensor(),
        download=False,
    )
    if max_samples is not None:
        dataset = Subset(dataset, range(min(max_samples, len(dataset))))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = create_resnet18().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    correct_count = 0
    incorrect_count = 0

    with GradCAM(model, model.layer4[-1]) as gradcam:
        for images, labels in loader:
            for image, label in zip(images, labels.view(-1)):
                if correct_count >= 10 and incorrect_count >= 10:
                    break
                input_image = image.unsqueeze(0).to(device)
                with torch.no_grad():
                    predicted = int(model(input_image).argmax(dim=1).item())
                is_correct = predicted == int(label)
                if is_correct and correct_count >= 10:
                    continue
                if not is_correct and incorrect_count >= 10:
                    continue
                cam = gradcam(input_image, predicted)[0, 0]
                category = "correct" if is_correct else "wrong"
                index = correct_count + 1 if is_correct else incorrect_count + 1
                filename = f"{category}_{index:02d}.png"
                save_overlay(
                    image,
                    cam,
                    output_dir / filename,
                    f"true: {CLASS_LABELS[int(label)]} | predicted: {CLASS_LABELS[predicted]}",
                )
                records.append(
                    {
                        "file": filename,
                        "correct": is_correct,
                        "true_class": int(label),
                        "predicted_class": predicted,
                        "true_label": CLASS_LABELS[int(label)],
                        "predicted_label": CLASS_LABELS[predicted],
                    }
                )
                if is_correct:
                    correct_count += 1
                else:
                    incorrect_count += 1
            if correct_count >= 10 and incorrect_count >= 10:
                break

    if correct_count < 10 or incorrect_count < 10:
        raise RuntimeError(
            f"Could not collect 10 correct and 10 incorrect predictions; "
            f"collected {correct_count} correct and {incorrect_count} incorrect."
        )
    create_analysis(records, checkpoint, analysis_path)
    manifest = {
        # Relative path so the manifest does not leak a local machine path.
        "checkpoint": _display_path(checkpoint),
        "device": str(device),
        "evaluated_examples": len(records),
        "correct_examples": correct_count,
        "incorrect_examples": incorrect_count,
        "records": records,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Day 4 ResNet18 Grad-CAM examples.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--analysis-path", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    generate_gradcam(**vars(args))


if __name__ == "__main__":
    main()
