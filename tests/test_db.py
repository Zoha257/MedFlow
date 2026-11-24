import uuid

import psycopg2
import pytest

from src.database import db

SUMMARY = {
    "train": {"num_samples": 89996, "image_height": 28, "image_width": 28, "channels": 3},
    "val": {"num_samples": 10004, "image_height": 28, "image_width": 28, "channels": 3},
    "test": {"num_samples": 7180, "image_height": 28, "image_width": 28, "channels": 3},
}


@pytest.fixture()
def temp_database(monkeypatch):
    """Create a throwaway database so the test never touches real data."""
    name = f"medflow_test_{uuid.uuid4().hex[:8]}"
    monkeypatch.setenv("POSTGRES_DB", name)
    monkeypatch.setattr(db, "_INITIALIZED", False)
    try:
        db.ensure_database_exists()  # also exercises CREATE DATABASE on a fresh server
    except psycopg2.OperationalError:
        pytest.skip("PostgreSQL is not available; start it and set POSTGRES_* in .env.")
    yield name
    monkeypatch.setenv("POSTGRES_DB", "postgres")
    connection = psycopg2.connect(**db.get_db_config())
    try:
        connection.autocommit = True  # DROP DATABASE cannot run inside a transaction
        with connection.cursor() as cursor:
            cursor.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
    finally:
        connection.close()


def _rows(query):
    with db.get_connection() as connection, connection.cursor() as cursor:
        cursor.execute(query)
        return cursor.fetchall()


def test_dataset_metadata_is_idempotent(temp_database) -> None:
    db.populate_dataset_metadata("PathMNIST", SUMMARY)
    db.populate_dataset_metadata("PathMNIST", SUMMARY)
    rows = _rows("SELECT split, num_samples FROM dataset_records ORDER BY split")
    assert rows == [("test", 7180), ("train", 89996), ("val", 10004)]


def test_legacy_duplicate_rows_are_removed(temp_database) -> None:
    db.init_database()
    with db.get_connection() as connection, connection.cursor() as cursor:
        cursor.execute("DROP INDEX dataset_records_dataset_split_key")
        for _ in range(3):
            cursor.execute(
                "INSERT INTO dataset_records (dataset_name, split, num_samples, image_height, image_width, channels) "
                "VALUES ('PathMNIST', 'train', 1, 28, 28, 3)"
            )
        connection.commit()
    db.init_database(force=True)
    assert _rows("SELECT count(*) FROM dataset_records") == [(1,)]


def test_processing_logs_record_per_split_entries(temp_database) -> None:
    db.record_processing_log("PathMNIST", "train", "download_and_preprocess", 10, 1.5)
    db.record_processing_log("PathMNIST", "val", "download_and_preprocess", 5, 0.5)
    assert _rows("SELECT split, processing_time_seconds FROM processing_logs ORDER BY split") == [
        ("train", 1.5),
        ("val", 0.5),
    ]
