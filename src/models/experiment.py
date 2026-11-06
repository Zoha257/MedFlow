from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import psycopg2
from dotenv import load_dotenv

from src.database.db import get_connection, init_database

ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")


def init_experiment_table() -> None:
    init_database()
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                ALTER TABLE experiments
                    ADD COLUMN IF NOT EXISTS learning_rate DOUBLE PRECISION,
                    ADD COLUMN IF NOT EXISTS batch_size INTEGER,
                    ADD COLUMN IF NOT EXISTS epochs INTEGER,
                    ADD COLUMN IF NOT EXISTS accuracy DOUBLE PRECISION,
                    ADD COLUMN IF NOT EXISTS precision_macro DOUBLE PRECISION,
                    ADD COLUMN IF NOT EXISTS recall_macro DOUBLE PRECISION,
                    ADD COLUMN IF NOT EXISTS f1_score DOUBLE PRECISION,
                    ADD COLUMN IF NOT EXISTS training_time_seconds DOUBLE PRECISION,
                    ADD COLUMN IF NOT EXISTS inference_time_seconds DOUBLE PRECISION,
                    ADD COLUMN IF NOT EXISTS num_parameters BIGINT
                """
            )
        connection.commit()


def record_experiment(
    experiment_name: str,
    model_name: str,
    augmentation: bool,
    metrics: dict[str, Any],
) -> None:
    init_experiment_table()
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO experiments
                    (
                        experiment_name, model_name, augmentation,
                        learning_rate, batch_size, epochs,
                        accuracy, precision_macro, recall_macro, f1_score,
                        training_time_seconds, inference_time_seconds,
                        num_parameters, metrics
                    )
                VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s::jsonb
                )
                """,
                (
                    experiment_name,
                    model_name,
                    augmentation,
                    metrics.get("learning_rate"),
                    metrics.get("batch_size"),
                    metrics.get("epochs"),
                    metrics.get("accuracy"),
                    metrics.get("precision_macro"),
                    metrics.get("recall_macro"),
                    metrics.get("f1_macro"),
                    metrics.get("training_time_seconds"),
                    metrics.get("inference_time_seconds"),
                    metrics.get("num_parameters"),
                    json.dumps(metrics),
                ),
            )
        connection.commit()
