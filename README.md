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
- `src/models/` — baseline CNN, training, and evaluation code
- `src/database/` — PostgreSQL setup and metadata insertion utilities
- `data/raw/` — raw dataset data from MedMNIST
- `data/processed/` — normalized and processed dataset outputs
- `results/` — evaluation metrics and confusion matrix

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

### Windows portable PostgreSQL setup

If PostgreSQL is installed with Scoop, initialize and start a user-owned server
without requiring administrator access:

```powershell
scoop install postgresql
$pg = "$env:USERPROFILE\scoop\apps\postgresql\current\bin"
$data = "$env:USERPROFILE\postgres-data-medflow"
& "$pg\initdb.exe" -D $data -U medflow --auth=trust
& "$pg\pg_ctl.exe" -D $data -l "$data\server.log" -o "-p 5432" start
& "$pg\psql.exe" -h localhost -p 5432 -U medflow -d postgres -c "CREATE DATABASE medflow;"
```

Then run the metadata pipeline:

```powershell
python -m src.database.db
```

## Day 2 baseline model

The Day 2 pipeline trains a deterministic PyTorch CNN baseline and records
accuracy, macro precision, macro recall, macro F1, training time, parameter
count, and a confusion matrix:

```powershell
python -m src.models.train --epochs 1 --batch-size 1024
```

Outputs:

- `models/baseline_model.pth`
- `results/results.json`
- `results/confusion_matrix.png`

The checked-in full-dataset run used one epoch, seed `42`, and CPU execution.
Its measured test accuracy was `0.6753` and macro F1 was `0.5923`. These are
reported measurements from the run, not target or fabricated values.

## Notes

- The project is intended for research/educational use only and is not a clinical diagnostic system.
- Dataset files are not stored in GitHub; the official MedMNIST distribution is used.
