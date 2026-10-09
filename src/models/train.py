from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from medmnist import PathMNIST
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from .cnn import BaselineCNN
from .evaluate import evaluate_model, save_evaluation_artifacts

ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def create_loader(split: str, batch_size: int, max_samples: int | None = None) -> DataLoader:
    dataset = PathMNIST(
        split=split,
        transform=transforms.ToTensor(),
        download=False,
    )
    if max_samples is not None:
        dataset = Subset(dataset, range(min(max_samples, len(dataset))))
    return DataLoader(dataset, batch_size=batch_size, shuffle=split == "train", num_workers=0)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0
    for images, labels in loader:
        targets = labels.view(-1).long().to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images.to(device)), targets)
        loss.backward()
        optimizer.step()
        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_items += batch_size
    return total_loss / total_items


def train_baseline(
    epochs: int = 1,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    seed: int = 42,
    max_samples: int | None = None,
) -> dict[str, object]:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = create_loader("train", batch_size, max_samples)
    val_loader = create_loader("val", batch_size, max_samples)
    test_loader = create_loader("test", batch_size, max_samples)

    model = BaselineCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    history: list[dict[str, float]] = []
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics, _ = evaluate_model(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": loss, **{f"val_{key}": value for key, value in val_metrics.items()}})
        print(f"epoch={epoch} train_loss={loss:.4f} val_accuracy={val_metrics['accuracy']:.4f}")

    test_metrics, matrix = evaluate_model(model, test_loader, device)
    training_seconds = time.perf_counter() - started
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = MODEL_DIR / "baseline_model.pth"
    torch.save(model.state_dict(), checkpoint)
    metrics = {
        **test_metrics,
        "model": "BaselineCNN",
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
        "device": str(device),
        "training_time_seconds": training_seconds,
        "num_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "history": history,
    }
    save_evaluation_artifacts(metrics, matrix, RESULTS_DIR)
    (RESULTS_DIR / "results.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved model to {checkpoint}")
    print(f"Saved results to {RESULTS_DIR}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the Day 2 PathMNIST baseline CNN.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None, help="Optional small run for smoke testing.")
    args = parser.parse_args()
    train_baseline(**vars(args))


if __name__ == "__main__":
    main()
