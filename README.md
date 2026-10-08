# MedFlow-XAI

This project implements the Day 1 milestone of the MedFlow-XAI handover: data acquisition, validation, preprocessing, and dataset metadata tracking for the PathMNIST dataset.

## Day 1 goal

The Day 1 success condition from the handover is:

- `python -m src.data.download` works
- the dataset is downloaded
- preprocessing works
- PostgreSQL contains dataset metadata
- the codebase is reproducible and committed

## Project structure

- `src/data/` — dataset download, validation, and preprocessing scripts
- `src/database/` — PostgreSQL setup and metadata insertion utilities
- `data/raw/` — raw dataset data from MedMNIST
- `data/processed/` — normalized and processed dataset outputs

## Quick start

1. Create and activate a virtual environment.
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure PostgreSQL in `.env` (copy from `.env.example`).
4. Start PostgreSQL locally.
5. Run:
   ```bash
   python -m src.data.download
   python -m src.data.validate
   python -m src.data.preprocess
   python -m src.database.db
   ```

## PostgreSQL configuration

Copy the example file and adjust values for your local database:

```bash
cp .env.example .env
```

Example variables:

```env
POSTGRES_DB=medflow
POSTGRES_USER=medflow
POSTGRES_PASSWORD=medflow
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
```

## Notes

- The project is intended for research/educational use only and is not a clinical diagnostic system.
- Dataset files are not stored in GitHub; the official MedMNIST distribution is used.
