import base64
import io

from fastapi.testclient import TestClient
from PIL import Image

from src.api.main import MODEL_VERSION, app


def image_bytes() -> io.BytesIO:
    image = Image.new("RGB", (28, 28), color=(120, 80, 160))
    output = io.BytesIO()
    image.save(output, format="PNG")
    output.seek(0)
    return output


def test_health_reports_loaded_model() -> None:
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["model"] == "ResNet18"
    assert payload["model_version"] == MODEL_VERSION


def test_predict_matches_handover_contract() -> None:
    response = TestClient(app).post(
        "/predict",
        files={"file": ("sample.png", image_bytes(), "image/png")},
    )
    assert response.status_code == 200
    payload = response.json()
    # Fields required by the handover document.
    assert isinstance(payload["prediction"], str)
    assert 0.0 <= payload["confidence"] <= 1.0
    assert payload["model_version"] == MODEL_VERSION
    # Extra detail used by the dashboard.
    assert payload["predicted_class"] in range(9)
    assert payload["predicted_label"] == payload["prediction"]
    assert len(payload["probabilities"]) == 9


def test_explain_matches_handover_contract() -> None:
    response = TestClient(app).post(
        "/explain",
        files={"file": ("sample.png", image_bytes(), "image/png")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload["prediction"], str)
    assert 0.0 <= payload["confidence"] <= 1.0
    assert payload["model_version"] == MODEL_VERSION
    assert payload["heatmap_size"] == {"width": 28, "height": 28}
    assert base64.b64decode(payload["heatmap"]).startswith(b"\x89PNG")
    # Backwards-compatible alias.
    assert payload["heatmap_base64"] == payload["heatmap"]


def test_rejects_non_image_upload() -> None:
    response = TestClient(app).post(
        "/predict",
        files={"file": ("sample.txt", io.BytesIO(b"not an image"), "text/plain")},
    )
    assert response.status_code == 415
