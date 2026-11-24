from pathlib import Path

from src.models.generate_gradcam import create_analysis


def record(name, true, pred):
    return {"file": name, "correct": true == pred, "true_label": true, "predicted_label": pred}


def test_analysis_warns_on_class_collapse(tmp_path: Path) -> None:
    records = [record("correct_01.png", "background", "background"),
               record("wrong_01.png", "mucus", "background")]
    out = tmp_path / "analysis.md"
    create_analysis(records, Path("models/resnet18.pth"), out)
    assert "class collapse" in out.read_text()


def test_analysis_makes_no_collapse_claim_for_varied_predictions(tmp_path: Path) -> None:
    records = [record("correct_01.png", "adipose", "adipose"),
               record("wrong_01.png", "mucus", "debris")]
    out = tmp_path / "analysis.md"
    create_analysis(records, Path("models/resnet18.pth"), out)
    text = out.read_text()
    assert "class collapse" not in text
    assert "Accuracy among selected" not in text  # selection-biased number must not appear
    assert "Manual review" in text
