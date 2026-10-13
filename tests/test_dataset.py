from medmnist import PathMNIST


def test_pathmnist_splits_have_expected_shape_and_classes() -> None:
    dataset = PathMNIST(split="test", download=False)
    image, label = dataset[0]

    assert len(dataset) == 7180
    assert image.size == (28, 28)
    assert image.mode == "RGB"
    assert int(label[0]) in range(9)
