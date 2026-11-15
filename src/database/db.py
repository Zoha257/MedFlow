from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import psycopg2
from dotenv import load_dotenv
from psycopg2 import sql

ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")


def get_db_config() -> Dict[str, str]:
    return {
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": os.getenv("POSTGRES_PORT", "5432"),
        "dbname": os.getenv("POSTGRES_DB", "medflow"),
        "user": os.getenv("POSTGRES_USER", "medflow"),
        "password": os.getenv("POSTGRES_PASSWORD", "medflow"),
    }


def get_connection(database_name: Optional[str] = None):
    config = get_db_config()
    if database_name is not None:
        config["dbname"] = database_name
    return psycopg2.connect(**config)


def ensure_database_exists() -> str:
    config = get_db_config()
    database_name = config["dbname"]

    with psycopg2.connect(
        host=config["host"],
        port=config["port"],
        dbname="postgres",
        user=config["user"],
        password=config["password"],
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s",
                (database_name,),
            )
            if cursor.fetchone() is None:
                cursor.execute(sql.SQL("CREATE DATABASE {}" ).format(sql.Identifier(database_name)))

    return database_name


def init_database() -> None:
    database_name = ensure_database_exists()

    with get_connection(database_name) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS dataset_records (
                    id SERIAL PRIMARY KEY,
                    dataset_name VARCHAR(255) NOT NULL,
                    split VARCHAR(50) NOT NULL,
                    num_samples INTEGER NOT NULL,
                    image_height INTEGER NOT NULL,
                    image_width INTEGER NOT NULL,
                    channels INTEGER NOT NULL,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS processing_logs (
                    id SERIAL PRIMARY KEY,
                    dataset_name VARCHAR(255) NOT NULL,
                    split VARCHAR(50) NOT NULL,
                    operation VARCHAR(255) NOT NULL,
                    samples_processed INTEGER NOT NULL,
                    processing_time_seconds DOUBLE PRECISION NOT NULL,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS experiments (
                    id SERIAL PRIMARY KEY,
                    experiment_name VARCHAR(255) NOT NULL,
                    model_name VARCHAR(255) NOT NULL,
                    augmentation BOOLEAN NOT NULL DEFAULT FALSE,
                    learning_rate DOUBLE PRECISION,
                    batch_size INTEGER,
                    epochs INTEGER,
                    accuracy DOUBLE PRECISION,
                    precision_macro DOUBLE PRECISION,
                    recall_macro DOUBLE PRECISION,
                    f1_score DOUBLE PRECISION,
                    training_time_seconds DOUBLE PRECISION,
                    inference_time_seconds DOUBLE PRECISION,
                    num_parameters BIGINT,
                    metrics JSONB NOT NULL DEFAULT '{}'::jsonb,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    id SERIAL PRIMARY KEY,
                    image_id VARCHAR(255),
                    predicted_class INTEGER NOT NULL,
                    confidence DOUBLE PRECISION NOT NULL,
                    model_version VARCHAR(255) NOT NULL,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.commit()

    print(f"PostgreSQL database initialized: {database_name}")


def record_dataset_metadata(
    dataset_name: str,
    split: str,
    num_samples: int,
    image_height: int,
    image_width: int,
    channels: int,
) -> None:
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO dataset_records (
                    dataset_name,
                    split,
                    num_samples,
                    image_height,
                    image_width,
                    channels
                ) VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (dataset_name, split, num_samples, image_height, image_width, channels),
            )
        connection.commit()


def populate_dataset_metadata(dataset_name: str, summary: Dict[str, Any]) -> None:
    init_database()

    for split_name, split_summary in summary.items():
        if split_name not in {"train", "val", "test"}:
            continue
        record_dataset_metadata(
            dataset_name=dataset_name,
            split=split_name,
            num_samples=int(split_summary["num_samples"]),
            image_height=int(split_summary["image_height"]),
            image_width=int(split_summary["image_width"]),
            channels=int(split_summary["channels"]),
        )

    print(f"Stored dataset metadata for {dataset_name} in PostgreSQL.")


def record_processing_log(
    dataset_name: str,
    split: str,
    operation: str,
    samples_processed: int,
    processing_time_seconds: float,
) -> None:
    init_database()
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO processing_logs (
                    dataset_name, split, operation, samples_processed,
                    processing_time_seconds
                ) VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    dataset_name,
                    split,
                    operation,
                    samples_processed,
                    processing_time_seconds,
                ),
            )
        connection.commit()


def record_prediction(
    predicted_class: int,
    confidence: float,
    model_version: str,
    image_id: Optional[str] = None,
) -> None:
    init_database()
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO predictions (
                    image_id, predicted_class, confidence, model_version
                ) VALUES (%s, %s, %s, %s)
                """,
                (image_id, predicted_class, confidence, model_version),
            )
        connection.commit()


def main() -> None:
    try:
        init_database()
        summary_path = ROOT_DIR / "data" / "processed" / "dataset_summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            populate_dataset_metadata("PathMNIST", summary)
        else:
            print(f"No dataset summary found at {summary_path}. Run the download pipeline first.")
    except Exception as exc:
        print("PostgreSQL is not available on the configured host.")
        print(f"Please start PostgreSQL and set POSTGRES_* values in .env. Details: {exc}")


if __name__ == "__main__":
    main()
