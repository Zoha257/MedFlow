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

## Day 3 ResNet18 experiments

Day 3 compares the measured Day 2 `BaselineCNN` run with two ResNet18
experiments: one without augmentation and one with horizontal/vertical flips
and 15-degree rotations applied to the training split. The script records each
run in MLflow, stores experiment metadata in PostgreSQL when it is available,
and writes the comparison to `results/model_comparison.json`.

Run the baseline first if `results/results.json` does not exist, then run:

```powershell
python -m src.models.train --epochs 1 --batch-size 1024
python -m src.models.train_day3 --epochs 1 --batch-size 1024 --experiment all
```

Use `--max-samples` only for a smoke test; those metrics must not be presented
as the full-dataset comparison. The generated comparison contains the actual
accuracy, macro F1, and training time for `BaselineCNN`, `ResNet18`, and
`ResNet18 + Aug`.

## Day 4 Grad-CAM explanations

Generate ten correct and ten incorrect ResNet18 explanations from the test
split after the full Day 3 checkpoint has been trained:

```powershell
python -m src.models.generate_gradcam
```

This creates `paper/figures/correct_01.png` through `correct_10.png`,
`wrong_01.png` through `wrong_10.png`, a `manifest.json`, and
`research/gradcam_analysis.md`. The report describes the observed examples
without treating Grad-CAM as clinical evidence.

## Day 5 FastAPI service

The Day 5 service exposes the trained ResNet18 model through:

- `GET /health` — model and device readiness
- `POST /predict` — class, confidence, and all class probabilities
- `POST /explain` — prediction plus a base64-encoded Grad-CAM overlay

Run locally:

```powershell
python -m uvicorn src.api.main:app --reload
```

The API expects an image upload in a multipart field named `file`. The service
resizes inputs to the PathMNIST 28x28 format and limits uploads to 10 MB.
It is a research/educational classifier, not a clinical diagnostic tool.

Run with Docker:

```powershell
docker compose up --build
```

The API is then available at `http://localhost:8000`; interactive API
documentation is available at `http://localhost:8000/docs`.

## Day 6 Streamlit application

Run the interactive frontend locally:

```powershell
python -m streamlit run src/app.py
```

The frontend supports the Day 6 flow:

```text
Upload image -> prediction -> probability chart -> Grad-CAM -> explanation
```

It uses the same ResNet18 checkpoint and preprocessing as the FastAPI service.
The API and Streamlit app are separate processes; run the API independently
when integrating them with another client.

## Day 7 research report

The research report is available at
[`research/report.md`](research/report.md). Generate the dataset distribution,
training curves, model comparison, and architecture figures with:

```powershell
python -m src.research.generate_artifacts
```

The report uses only measured artifacts and explicitly labels the current
ResNet results as smoke runs. It does not claim clinical utility.

## Notes

- The project is intended for research/educational use only and is not a clinical diagnostic system.
- Dataset files are not stored in GitHub; the official MedMNIST distribution is used.
