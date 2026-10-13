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
                CREATE TABLE IF NOT EXISTS experiments (
                    id SERIAL PRIMARY KEY,
                    experiment_name VARCHAR(255) NOT NULL,
                    model_name VARCHAR(255) NOT NULL,
                    augmentation BOOLEAN NOT NULL,
                    metrics JSONB NOT NULL,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
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
                    (experiment_name, model_name, augmentation, metrics)
                VALUES (%s, %s, %s, %s::jsonb)
                """,
                (experiment_name, model_name, augmentation, json.dumps(metrics)),
            )
        connection.commit()
