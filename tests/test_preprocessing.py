import torch
from PIL import Image

from src.api.main import PREPROCESS


def test_preprocessing_produces_model_input_tensor() -> None:
    image = Image.new("RGB", (64, 40), color=(100, 120, 140))
    tensor = PREPROCESS(image)

    assert tensor.shape == (3, 28, 28)
    assert tensor.dtype == torch.float32
    assert 0.0 <= float(tensor.min()) <= float(tensor.max()) <= 1.0
