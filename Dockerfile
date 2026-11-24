FROM python:3.13-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY models/resnet18.pth ./models/resnet18.pth
# The Streamlit pages read these saved artifacts at runtime.
COPY results ./results
COPY paper/figures ./paper/figures
COPY data/processed/dataset_summary.json ./data/processed/dataset_summary.json

EXPOSE 8000 8501

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
