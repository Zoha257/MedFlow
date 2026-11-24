import pytest

from src.research.render_results import START, END, build_table, build_warnings, replace_block, run_scope

FULL = {"accuracy": 0.9, "f1_macro": 0.88, "training_time_seconds": 10.0, "num_parameters": 5,
        "epochs": 10, "max_samples": None, "num_predicted_classes": 9}


def test_run_scope_distinguishes_full_smoke_and_unrecorded() -> None:
    assert run_scope({"max_samples": None}) == "full dataset"
    assert "SMOKE" in run_scope({"max_samples": 32})
    assert run_scope({}) == "not recorded"


def test_table_numbers_come_straight_from_the_metrics() -> None:
    table = build_table({"resnet18": FULL})
    assert "| ResNet18 | 0.9000 | 0.8800 | 10.0 s | n/a | 5 | 10 | full dataset |" in table


def test_no_warnings_for_clean_equal_budget_runs() -> None:
    assert build_warnings({"baseline_cnn": FULL, "resnet18": dict(FULL)}) == []


def test_warnings_flag_smoke_collapse_and_unequal_epochs() -> None:
    smoke = {**FULL, "max_samples": 32, "num_predicted_classes": 1, "epochs": 1}
    text = " ".join(build_warnings({"baseline_cnn": FULL, "resnet18": smoke}))
    assert "smoke run" in text and "class collapse" in text and "like-for-like" in text


def test_replace_block_swaps_only_the_marked_region() -> None:
    doc = f"before\n{START}\nold\n{END}\nafter"
    out = replace_block(doc, f"{START}\nnew\n{END}")
    assert out == f"before\n{START}\nnew\n{END}\nafter"
    with pytest.raises(ValueError):
        replace_block("no markers", "x")
