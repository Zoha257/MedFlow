from __future__ import annotations

import base64
import io
import os
from functools import lru_cache
from pathlib import Path
from typing import Annotated

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from src.models.gradcam import GradCAM
from src.models.resnet import create_resnet18

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = ROOT_DIR / "models" / "resnet18.pth"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
CLASS_LABELS = {
    0: "adipose",
    1: "background",
    2: "debris",
    3: "lymphocytes",
    4: "mucus",
    5: "smooth muscle",
    6: "normal colon mucosa",
    7: "cancer-associated stroma",
    8: "colorectal adenocarcinoma epithelium",
}
PREPROCESS = transforms.Compose(
    [
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ]
)

app = FastAPI(
    title="MedFlow-XAI API",
    description="Research/educational PathMNIST prediction and explanation service.",
    version="1.0.0",
)


class ModelService:
    def __init__(self, checkpoint: Path) -> None:
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found at {checkpoint}. "
                "Train the Day 3 ResNet18 model before starting the API."
            )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = create_resnet18().to(self.device)
        state = torch.load(checkpoint, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()

    def tensor_from_image(self, image: Image.Image) -> torch.Tensor:
        return PREPROCESS(image.convert("RGB")).unsqueeze(0).to(self.device)

    def predict_tensor(self, tensor: torch.Tensor) -> tuple[int, torch.Tensor]:
        with torch.inference_mode():
            probabilities = torch.softmax(self.model(tensor), dim=1)[0]
        return int(probabilities.argmax().item()), probabilities

    def predict(self, image: Image.Image) -> dict[str, object]:
        tensor = self.tensor_from_image(image)
        predicted_class, probabilities = self.predict_tensor(tensor)
        return {
            "predicted_class": predicted_class,
            "predicted_label": CLASS_LABELS[predicted_class],
            "confidence": float(probabilities[predicted_class].item()),
            "probabilities": {
                CLASS_LABELS[index]: float(probability.item())
                for index, probability in enumerate(probabilities)
            },
            "device": str(self.device),
        }

    def explain(self, image: Image.Image) -> dict[str, object]:
        tensor = self.tensor_from_image(image)
        predicted_class, probabilities = self.predict_tensor(tensor)
        with GradCAM(self.model, self.model.layer4[-1]) as gradcam:
            heatmap = gradcam(tensor, predicted_class)[0, 0].detach().cpu()
        image_array = (
            tensor[0]
            .detach()
            .cpu()
            .permute(1, 2, 0)
            .numpy()
            * 255
        ).clip(0, 255).astype("uint8")
        heatmap_array = heatmap.numpy()
        overlay = image_array.copy()
        overlay[:, :, 0] = np_clip_uint8(image_array[:, :, 0] * 0.55 + heatmap_array * 255 * 0.45)
        overlay[:, :, 1] = np_clip_uint8(image_array[:, :, 1] * 0.55)
        overlay[:, :, 2] = np_clip_uint8(image_array[:, :, 2] * 0.55)
        output = io.BytesIO()
        Image.fromarray(overlay).save(output, format="PNG")
        return {
            "predicted_class": predicted_class,
            "predicted_label": CLASS_LABELS[predicted_class],
            "confidence": float(probabilities[predicted_class].item()),
            "heatmap_base64": base64.b64encode(output.getvalue()).decode("ascii"),
            "heatmap_size": {"width": 28, "height": 28},
            "device": str(self.device),
        }


def np_clip_uint8(values: np.ndarray) -> np.ndarray:
    return np.clip(values, 0, 255).astype("uint8")


@lru_cache(maxsize=1)
def get_model_service() -> ModelService:
    configured_checkpoint = os.getenv("MEDFLOW_MODEL_CHECKPOINT")
    checkpoint = Path(configured_checkpoint) if configured_checkpoint else DEFAULT_CHECKPOINT
    return ModelService(checkpoint)


async def read_image(upload: UploadFile) -> Image.Image:
    if upload.content_type is not None and not upload.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Upload must be an image file.")
    contents = await upload.read(MAX_UPLOAD_BYTES + 1)
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image upload exceeds the 10 MB limit.")
    try:
        image = Image.open(io.BytesIO(contents))
        image.load()
        return image
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a readable image.") from exc


def service_or_error() -> ModelService:
    try:
        return get_model_service()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/health")
def health() -> dict[str, str]:
    service = service_or_error()
    return {"status": "ok", "model": "ResNet18", "device": str(service.device)}


@app.post("/predict")
async def predict(file: Annotated[UploadFile, File(...)]) -> dict[str, object]:
    image = await read_image(file)
    return service_or_error().predict(image)


@app.post("/explain")
async def explain(file: Annotated[UploadFile, File(...)]) -> dict[str, object]:
    image = await read_image(file)
    return service_or_error().explain(image)
