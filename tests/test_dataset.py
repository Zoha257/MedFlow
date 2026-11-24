import pytest
from medmnist import PathMNIST


def test_pathmnist_splits_have_expected_shape_and_classes() -> None:
    try:
        dataset = PathMNIST(split="test", download=False)
    except (RuntimeError, FileNotFoundError):
        pytest.skip("PathMNIST is not downloaded; run `python -m src.data.download` first.")
    image, label = dataset[0]

    assert len(dataset) == 7180
    assert image.size == (28, 28)
    assert image.mode == "RGB"
    assert int(label[0]) in range(9)
