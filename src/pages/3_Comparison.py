from __future__ import annotations

import pandas as pd
import streamlit as st

from src.research.render_results import build_warnings
from src.ui import RESULTS_DIR, apply_theme, load_json, render_disclaimer

st.set_page_config(page_title="Comparison · MedFlow-XAI", page_icon="✦", layout="wide")
apply_theme()
st.title("CNN vs ResNet18")
st.write("A comparison of the saved experiments. Warnings below are computed from the recorded run scope, epochs and predicted-class counts.")

comparison = load_json(RESULTS_DIR / "model_comparison.json")
if not comparison:
    st.warning("No model comparison artifact found.")
    st.stop()

for warning in build_warnings(comparison):
    st.warning(warning)

rows = []
for name, metrics in comparison.items():
    rows.append(
        {
            "Model": name.replace("_", " ").title(),
            "Accuracy": metrics.get("accuracy", 0),
            "Precision": metrics.get("precision_macro", 0),
            "Recall": metrics.get("recall_macro", 0),
            "F1-score": metrics.get("f1_macro", 0),
            "Epochs": metrics.get("epochs", "n/a"),
        }
    )
frame = pd.DataFrame(rows).set_index("Model")
st.dataframe(frame.style.format({key: "{:.1%}" for key in ["Accuracy", "Precision", "Recall", "F1-score"]}), use_container_width=True)
st.markdown("### Metric comparison")
st.bar_chart(frame[["Accuracy", "Precision", "Recall", "F1-score"]])

history_rows = []
for name, metrics in comparison.items():
    for point in metrics.get("history", []):
        history_rows.append({"Model": name.replace("_", " ").title(), **point})
if history_rows:
    st.markdown("### Training history")
    history = pd.DataFrame(history_rows)
    st.line_chart(history.pivot(index="epoch", columns="Model", values="val_accuracy"))

render_disclaimer()
