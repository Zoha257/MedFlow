import json
import time
from pathlib import Path

from medmnist import PathMNIST
from torchvision import transforms

from src.database.db import populate_dataset_metadata, record_processing_log

ROOT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT_DIR / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = ("train", "val", "test")


def download_dataset():
    print("Downloading PathMNIST...")

    summary = {}
    timings = {}
    for split in SPLITS:
        # Time each split on its own so processing_logs reports a meaningful
        # per-split duration instead of the same total repeated three times.
        started = time.perf_counter()
        dataset = PathMNIST(
            split=split,
            transform=transforms.ToTensor(),
            download=True,
        )
        first_image = dataset[0][0]
        summary[split] = {
            "num_samples": len(dataset),
            "image_height": first_image.shape[-2],
            "image_width": first_image.shape[-1],
            "channels": first_image.shape[0],
        }
        timings[split] = time.perf_counter() - started

    summary_path = OUTPUT_DIR / "dataset_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nPathMNIST downloaded successfully!")
    for split in SPLITS:
        print(f"{split:<5} samples: {summary[split]['num_samples']}")
    print(f"Dataset summary saved to: {summary_path}")

    try:
        # Idempotent: re-running updates the existing (dataset, split) rows.
        populate_dataset_metadata("PathMNIST", summary)
        for split, split_summary in summary.items():
            record_processing_log(
                dataset_name="PathMNIST",
                split=split,
                operation="download_and_preprocess",
                samples_processed=int(split_summary["num_samples"]),
                processing_time_seconds=timings[split],
            )
    except Exception as exc:  # pragma: no cover - runtime-only fallback when PostgreSQL is unavailable.
        print(f"PostgreSQL metadata storage skipped: {exc}")
        print("If PostgreSQL is running locally, set POSTGRES_* values in .env and rerun python -m src.database.db")


if __name__ == "__main__":
    download_dataset()
