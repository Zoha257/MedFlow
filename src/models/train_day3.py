from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import psycopg2
import torch
from medmnist import PathMNIST
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from .evaluate import evaluate_model, save_evaluation_artifacts
from .experiment import record_experiment
from .resnet import create_resnet18
from .train import train_one_epoch

ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"
MLFLOW_DB = ROOT_DIR / "mlflow.db"
BASELINE_RESULTS = RESULTS_DIR / "results.json"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def create_resnet_loader(
    split: str,
    batch_size: int,
    augmentation: bool,
    max_samples: int | None,
) -> DataLoader:
    transform_steps = []
    if split == "train" and augmentation:
        transform_steps.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(15),
            ]
        )
    transform_steps.append(transforms.ToTensor())
    dataset = PathMNIST(
        split=split,
        transform=transforms.Compose(transform_steps),
        download=False,
    )
    if max_samples is not None:
        dataset = Subset(dataset, range(min(max_samples, len(dataset))))
    return DataLoader(dataset, batch_size=batch_size, shuffle=split == "train", num_workers=0)


def run_experiment(
    name: str,
    augmentation: bool,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    max_samples: int | None = None,
) -> dict[str, Any]:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = create_resnet_loader("train", batch_size, augmentation, max_samples)
    val_loader = create_resnet_loader("val", batch_size, False, max_samples)
    test_loader = create_resnet_loader("test", batch_size, False, max_samples)
    model = create_resnet18().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    history: list[dict[str, float]] = []
    started = time.perf_counter()
    best_f1, best_epoch, best_state = -1.0, 0, None
    # Smoke runs (--max-samples) must never overwrite the checkpoint the API serves.
    run_name = f"{name}_smoke" if max_samples is not None else name

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "model": "ResNet18",
                "augmentation": augmentation,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "seed": seed,
                "max_samples": max_samples,
            }
        )
        for epoch in range(1, epochs + 1):
            loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_metrics, _ = evaluate_model(model, val_loader, device)
            epoch_metrics = {
                "epoch": epoch,
                "train_loss": loss,
                **{f"val_{key}": value for key, value in val_metrics.items()},
            }
            history.append(epoch_metrics)
            mlflow.log_metrics({key: float(value) for key, value in epoch_metrics.items()}, step=epoch)
            print(f"{name} epoch={epoch} train_loss={loss:.4f} val_accuracy={val_metrics['accuracy']:.4f}")
            if val_metrics["f1_macro"] > best_f1:  # model selection uses the VALIDATION split only
                best_f1, best_epoch = val_metrics["f1_macro"], epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        model.load_state_dict(best_state)  # test the best-validation epoch, not simply the last one
        print(f"{name}: selected epoch {best_epoch} (best validation macro-F1 {best_f1:.4f})")
        test_metrics, matrix = evaluate_model(model, test_loader, device)
        training_seconds = time.perf_counter() - started
        metrics: dict[str, Any] = {
            **test_metrics,
            "model": "ResNet18",
            "augmentation": augmentation,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "seed": seed,
            "max_samples": max_samples,  # None = full dataset
            "checkpoint_selection": "best validation macro-F1",
            "best_epoch": best_epoch,
            "device": str(device),
            "training_time_seconds": training_seconds,
            "num_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "history": history,
        }
        mlflow.log_metrics({key: value for key, value in test_metrics.items()})
        mlflow.log_metric("training_time_seconds", training_seconds)
        mlflow.log_metric("num_parameters", metrics["num_parameters"])

    checkpoint = MODEL_DIR / f"{run_name}.pth"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint)
    output_dir = RESULTS_DIR / run_name
    if test_metrics.get("num_predicted_classes", 9) < 9:
        print(
            f"WARNING: {run_name} predicted only {test_metrics['num_predicted_classes']} of 9 classes "
            "on the test split. Treat this run as failed/collapsed, not as a result."
        )
    save_evaluation_artifacts(metrics, matrix, output_dir)
    (output_dir / "results.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    try:
        record_experiment(run_name, "ResNet18", augmentation, metrics)
    except psycopg2.Error as exc:
        print(f"PostgreSQL experiment logging skipped: {exc}")
    return metrics


def load_baseline_metrics() -> dict[str, Any]:
    """Load the measured Day 2 metrics used in the model comparison."""
    if not BASELINE_RESULTS.exists():
        raise FileNotFoundError(
            f"Baseline results are required at {BASELINE_RESULTS}. "
            "Run `python -m src.models.train` before Day 3."
        )

    metrics = json.loads(BASELINE_RESULTS.read_text(encoding="utf-8"))
    required = {"accuracy", "f1_macro", "training_time_seconds", "model"}
    missing = required.difference(metrics)
    if missing:
        missing_names = ", ".join(sorted(missing))
        raise ValueError(f"Baseline results are missing required fields: {missing_names}")
    return metrics


def log_baseline_experiment(metrics: dict[str, Any]) -> None:
    """Record the existing Day 2 run alongside the Day 3 MLflow runs."""
    with mlflow.start_run(run_name="baseline_cnn"):
        params = {
            "model": metrics["model"],
            "augmentation": False,
            "epochs": metrics.get("epochs"),
            "batch_size": metrics.get("batch_size"),
            "learning_rate": metrics.get("learning_rate"),
            "seed": metrics.get("seed"),
        }
        mlflow.log_params({key: value for key, value in params.items() if value is not None})
        mlflow.log_metrics(
            {
                key: float(metrics[key])
                for key in ("accuracy", "precision_macro", "recall_macro", "f1_macro")
                if key in metrics
            }
        )
        if "training_time_seconds" in metrics:
            mlflow.log_metric("training_time_seconds", float(metrics["training_time_seconds"]))
        if "num_parameters" in metrics:
            mlflow.log_metric("num_parameters", float(metrics["num_parameters"]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Day 3 ResNet18 experiments.")
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of full training epochs. Use --epochs 1 only for a smoke test.",
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--experiment", choices=["resnet18", "resnet18_aug", "all"], default="all")
    args = parser.parse_args()

    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB.as_posix()}")
    mlflow.set_experiment("MedFlow-XAI Day 3")
    choices = (
        ["resnet18", "resnet18_aug"] if args.experiment == "all" else [args.experiment]
    )
    baseline_metrics = load_baseline_metrics()
    if baseline_metrics.get("epochs") != args.epochs:
        print(
            f"WARNING: baseline was trained for {baseline_metrics.get('epochs')} epoch(s) but ResNet18 "
            f"runs use {args.epochs}. Retrain the baseline with the same --epochs for a like-for-like comparison."
        )
    log_baseline_experiment(baseline_metrics)
    comparison: dict[str, dict[str, Any]] = {"baseline_cnn": baseline_metrics}
    for choice in choices:
        comparison[choice] = run_experiment(
            name=choice,
            augmentation=choice == "resnet18_aug",
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
            max_samples=args.max_samples,
        )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    comparison_path = RESULTS_DIR / (
        "model_comparison_smoke.json" if args.max_samples is not None else "model_comparison.json"
    )
    comparison_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(f"Saved comparison to {comparison_path}")


if __name__ == "__main__":
    main()
