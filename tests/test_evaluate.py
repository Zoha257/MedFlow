import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.cnn import BaselineCNN
from src.models.evaluate import evaluate_model


def test_evaluate_model_returns_required_metrics() -> None:
    model = BaselineCNN()
    images = torch.rand(6, 3, 28, 28)
    labels = torch.tensor([[0], [1], [2], [3], [4], [5]])
    metrics, matrix = evaluate_model(model, DataLoader(TensorDataset(images, labels), batch_size=2), torch.device("cpu"))

    assert {"accuracy", "precision_macro", "recall_macro", "f1_macro"} <= metrics.keys()
    assert matrix.shape == (9, 9)
    assert np.isfinite(list(metrics.values())).all()
