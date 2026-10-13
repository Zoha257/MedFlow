import torch

from src.models.gradcam import GradCAM
from src.models.resnet import create_resnet18


def test_gradcam_returns_normalized_input_sized_map() -> None:
    model = create_resnet18()
    model.eval()
    with GradCAM(model, model.layer4[-1]) as gradcam:
        heatmap = gradcam(torch.randn(1, 3, 28, 28), target_class=0)

    assert heatmap.shape == (1, 1, 28, 28)
    assert float(heatmap.detach().min()) >= 0.0
    assert float(heatmap.detach().max()) <= 1.0
