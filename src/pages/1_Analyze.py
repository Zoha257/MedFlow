from __future__ import annotations

import pandas as pd
import streamlit as st

from src.ui import (
    add_history,
    apply_theme,
    explanation_image,
    image_from_bytes,
    image_from_upload,
    load_service,
    report_bytes,
    render_disclaimer,
    sample_images,
)

st.set_page_config(page_title="Analyze · MedFlow-XAI", page_icon="✦", layout="wide")
apply_theme()
st.title("Analyze an image")
st.write("Upload your own tissue patch, or select a built-in sample to try the workflow immediately.")

upload, demo = st.columns([1.2, 0.8], gap="large")
with upload:
    uploaded = st.file_uploader("Upload PNG, JPG, JPEG, or BMP", type=["png", "jpg", "jpeg", "bmp"])
with demo:
    options = sample_images()
    selected = st.selectbox("Or choose a demo image", ["None"] + [name for name, path in options])

image = None
source = "upload"
if uploaded is not None:
    image = image_from_upload(uploaded)
elif selected != "None":
    selected_path = dict(options)[selected]
    if selected_path.exists():
        image = image_from_bytes(selected_path.read_bytes())
    source = selected

if image is None:
    st.info("Choose an upload or demo sample to generate a prediction.")
    render_disclaimer()
    st.stop()

try:
    service = load_service()
    prediction = service.predict(image)
    add_history(prediction, source)
    st.markdown("### Prediction")
    metric_columns = st.columns(3)
    for column, label, value in zip(
        metric_columns,
        ["Predicted class", "Confidence", "Inference device"],
        [prediction["predicted_label"], f"{float(prediction['confidence']):.1%}", prediction["device"]],
    ):
        with column:
            st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>', unsafe_allow_html=True)

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("### Original image")
        st.image(image, use_container_width=True)
    with right:
        st.markdown("### Grad-CAM heatmap")
        with st.spinner("Generating explanation..."):
            explanation = service.explain(image)
        st.image(explanation_image(explanation), use_container_width=True)
        st.caption("Highlighted regions contributed to the predicted class score.")

    probabilities = pd.DataFrame(
        {"Class": list(prediction["probabilities"]), "Probability": list(prediction["probabilities"].values())}
    ).set_index("Class")
    st.markdown("### Class probabilities")
    st.bar_chart(probabilities)
    st.download_button(
        "Download prediction report (CSV)",
        data=report_bytes(prediction),
        file_name="medflow_prediction_report.csv",
        mime="text/csv",
    )
except (FileNotFoundError, RuntimeError, ValueError) as exc:
    st.error(f"Model service unavailable: {exc}")

render_disclaimer()
