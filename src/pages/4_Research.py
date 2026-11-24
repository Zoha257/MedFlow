from __future__ import annotations

import streamlit as st

from src.ui import FIGURES_DIR, apply_theme, render_disclaimer

st.set_page_config(page_title="Research · MedFlow-XAI", page_icon="✦", layout="wide")
apply_theme()
st.title("Research context")
st.write("Architecture, reproducibility notes, and limitations behind the dashboard.")

left, right = st.columns([1.1, 0.9], gap="large")
with left:
    st.markdown("### Pipeline")
    st.markdown("`PathMNIST → preprocessing → ResNet18 → evaluation → Grad-CAM`")
    st.markdown("### Model")
    st.write("The classifier uses a ResNet18 adapted for 28 × 28 RGB images, with nine PathMNIST tissue classes and a task-specific final layer.")
    st.markdown("### Dataset")
    st.write("PathMNIST contains 9 tissue classes of histopathology patches. Images are resized to 28 × 28 and normalized as tensors for inference.")
    st.markdown("### Reproducibility")
    st.write("Training scripts support deterministic seeds, augmentation, MLflow tracking, saved checkpoints, JSON metrics, and confusion-matrix artifacts.")
with right:
    architecture = FIGURES_DIR / "architecture.png"
    if architecture.exists():
        st.image(str(architecture), caption="MedFlow-XAI architecture", use_container_width=True)

st.markdown("### Research limitations")
st.markdown(
    """
    - Results are only meaningful when the run scope is the full dataset and models share a training budget; the Comparison page shows warnings computed from the saved files.
    - PathMNIST is a curated research dataset and does not represent clinical deployment conditions or population diversity.
    - Confidence scores are model outputs, not calibrated clinical probabilities.
    - Grad-CAM is an attribution aid; highlighted regions do not prove causal or clinically meaningful reasoning.
    - This project has no prospective validation, external test cohort, clinical review, or regulatory clearance.
    """
)
render_disclaimer()
