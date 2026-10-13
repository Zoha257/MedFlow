import torch

from src.models.resnet import create_resnet18


def test_resnet18_output_shape() -> None:
    model = create_resnet18()
    outputs = model(torch.randn(2, 3, 28, 28))
    assert outputs.shape == (2, 9)
