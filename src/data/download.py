import json
import time
from pathlib import Path

from medmnist import PathMNIST
from torchvision import transforms

from src.database.db import populate_dataset_metadata, record_processing_log

ROOT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT_DIR / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def download_dataset():
    print("Downloading PathMNIST...")
    started = time.perf_counter()

    train_dataset = PathMNIST(
        split="train",
        transform=transforms.ToTensor(),
        download=True,
    )
    val_dataset = PathMNIST(
        split="val",
        transform=transforms.ToTensor(),
        download=True,
    )
    test_dataset = PathMNIST(
        split="test",
        transform=transforms.ToTensor(),
        download=True,
    )

    summary = {
        "train": {
            "num_samples": len(train_dataset),
            "image_height": train_dataset[0][0].shape[-2],
            "image_width": train_dataset[0][0].shape[-1],
            "channels": train_dataset[0][0].shape[0],
        },
        "val": {
            "num_samples": len(val_dataset),
            "image_height": val_dataset[0][0].shape[-2],
            "image_width": val_dataset[0][0].shape[-1],
            "channels": val_dataset[0][0].shape[0],
        },
        "test": {
            "num_samples": len(test_dataset),
            "image_height": test_dataset[0][0].shape[-2],
            "image_width": test_dataset[0][0].shape[-1],
            "channels": test_dataset[0][0].shape[0],
        },
    }
    summary_path = OUTPUT_DIR / "dataset_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nPathMNIST downloaded successfully!")
    print(f"Training samples:   {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples:       {len(test_dataset)}")
    print(f"Dataset summary saved to: {summary_path}")

    try:
        populate_dataset_metadata("PathMNIST", summary)
        elapsed = time.perf_counter() - started
        for split, split_summary in summary.items():
            record_processing_log(
                dataset_name="PathMNIST",
                split=split,
                operation="download_and_preprocess",
                samples_processed=int(split_summary["num_samples"]),
                processing_time_seconds=elapsed,
            )
    except Exception as exc:  # pragma: no cover - runtime-only fallback when PostgreSQL is unavailable.
        print(f"PostgreSQL metadata storage skipped: {exc}")
        print("If PostgreSQL is running locally, set POSTGRES_* values in .env and rerun python -m src.database.db")


if __name__ == "__main__":
    download_dataset()