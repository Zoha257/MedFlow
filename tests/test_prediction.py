from PIL import Image

from src.api.main import CLASS_LABELS, get_model_service


def test_prediction_contains_all_pathmnist_classes() -> None:
    prediction = get_model_service().predict(Image.new("RGB", (28, 28), color="white"))

    assert prediction["predicted_class"] in range(9)
    assert prediction["predicted_label"] in CLASS_LABELS.values()
    assert len(prediction["probabilities"]) == 9
    assert abs(sum(prediction["probabilities"].values()) - 1.0) < 1e-5
