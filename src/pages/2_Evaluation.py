from __future__ import annotations

import streamlit as st

from src.ui import RESULTS_DIR, apply_theme, load_json, render_disclaimer

st.set_page_config(page_title="Evaluation · MedFlow-XAI", page_icon="✦", layout="wide")
apply_theme()
st.title("Model evaluation")
st.write("Saved test-set metrics for the trained ResNet18 checkpoint.")

result = load_json(RESULTS_DIR / "resnet18" / "results.json")
if not result:
    st.warning("No evaluation artifact found. Run the training/evaluation command first.")
    st.stop()

metrics = [
    ("Accuracy", "accuracy"),
    ("Precision", "precision_macro"),
    ("Recall", "recall_macro"),
    ("F1-score", "f1_macro"),
]
columns = st.columns(4)
for column, (label, key) in zip(columns, metrics):
    with column:
        st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{float(result[key]):.1%}</div></div>', unsafe_allow_html=True)

st.markdown("### Confusion matrix")
matrix_path = RESULTS_DIR / "resnet18" / "confusion_matrix.png"
if matrix_path.exists():
    st.image(str(matrix_path), caption="ResNet18 test-set confusion matrix", use_container_width=True)
else:
    st.info("Confusion matrix image is not available yet.")
st.caption(f"Training epochs: {result.get('epochs', 'n/a')} · Inference time: {float(result.get('inference_time_seconds', 0)):.2f}s")
render_disclaimer()
