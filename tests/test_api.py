import io
import base64

from fastapi.testclient import TestClient
from PIL import Image

from src.api.main import app


def image_bytes() -> io.BytesIO:
    image = Image.new("RGB", (28, 28), color=(120, 80, 160))
    output = io.BytesIO()
    image.save(output, format="PNG")
    output.seek(0)
    return output


def test_health_reports_loaded_model() -> None:
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json()["model"] == "ResNet18"


def test_predict_returns_probabilities() -> None:
    response = TestClient(app).post(
        "/predict",
        files={"file": ("sample.png", image_bytes(), "image/png")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["predicted_class"] in range(9)
    assert len(payload["probabilities"]) == 9


def test_explain_returns_png_heatmap() -> None:
    response = TestClient(app).post(
        "/explain",
        files={"file": ("sample.png", image_bytes(), "image/png")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["heatmap_size"] == {"width": 28, "height": 28}
    assert base64.b64decode(payload["heatmap_base64"]).startswith(b"\x89PNG")


def test_rejects_non_image_upload() -> None:
    response = TestClient(app).post(
        "/predict",
        files={"file": ("sample.txt", io.BytesIO(b"not an image"), "text/plain")},
    )
    assert response.status_code == 415
