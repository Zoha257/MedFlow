import torch

from src.models.cnn import BaselineCNN


def test_baseline_cnn_output_shape() -> None:
    model = BaselineCNN(num_classes=9)
    outputs = model(torch.randn(4, 3, 28, 28))
    assert outputs.shape == (4, 9)


def test_baseline_cnn_has_trainable_parameters() -> None:
    model = BaselineCNN()
    assert sum(parameter.numel() for parameter in model.parameters()) > 0
