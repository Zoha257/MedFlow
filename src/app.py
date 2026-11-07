
from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

# Ensure the project root is available for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st
from PIL import Image

from src.api.main import ModelService, get_model_service


st.set_page_config(page_title="MedFlow-XAI", page_icon="✦", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    :root { --ink:#0b1220; --muted:#647084; --line:#e5e9ee; --mint:#c7f7df; --blue:#4d6fff; }
    .stApp { background:#f6f8fa; color:var(--ink); }
    .block-container { max-width:1180px; padding:4.25rem 2.5rem 4rem; }
    h1,h2,h3,p,div,button,label { font-family:'DM Sans',sans-serif; }
    h1,h2,h3 { font-family:'Space Grotesk',sans-serif; letter-spacing:-.04em; }
    .topbar { align-items:center; display:flex; justify-content:space-between; padding:.35rem 0 1.8rem; }
    .brand { align-items:center; display:flex; gap:.7rem; }
    .brand-mark { align-items:center; background:var(--ink); border-radius:12px; color:var(--mint); display:flex; height:38px; justify-content:center; width:38px; }
    .brand-name { font-family:'Space Grotesk'; font-size:1.05rem; font-weight:700; }
    .brand-name span { color:var(--blue); }
    .topbar-note { color:var(--muted); font-size:.78rem; letter-spacing:.06em; text-transform:uppercase; }
    .hero { background:var(--ink); border-radius:28px; color:white; overflow:hidden; padding:clamp(2rem,6vw,5.5rem) clamp(1.5rem,6vw,6rem); position:relative; }
    .hero:after { background:radial-gradient(circle,rgba(199,247,223,.32),transparent 68%); content:''; height:520px; position:absolute; right:-150px; top:-240px; width:520px; }
    .eyebrow { color:var(--mint); font-size:.72rem; font-weight:700; letter-spacing:.16em; position:relative; text-transform:uppercase; z-index:1; }
    .hero h1 { font-size:clamp(2.8rem,7vw,6.8rem); line-height:.94; margin:1.35rem 0; max-width:820px; position:relative; z-index:1; }
    .hero h1 em { color:var(--mint); font-style:normal; }
    .hero-copy { color:#b8c3d0; font-size:clamp(1rem,1.5vw,1.18rem); line-height:1.65; max-width:525px; position:relative; z-index:1; }
    .hero-pills { display:flex; flex-wrap:wrap; gap:.65rem; margin-top:2rem; position:relative; z-index:1; }
    .hero-pill { border:1px solid rgba(255,255,255,.16); border-radius:999px; color:#dce4ed; font-size:.78rem; padding:.55rem .85rem; }
    .section-kicker { color:var(--blue); font-size:.72rem; font-weight:700; letter-spacing:.15em; margin:3.5rem 0 .75rem; text-transform:uppercase; }
    .section-title { font-size:clamp(1.9rem,3.5vw,3rem); margin:0 0 .7rem; }
    .section-copy { color:var(--muted); line-height:1.65; max-width:650px; }
    .feature-card,.upload-card,.result-card { background:white; border:1px solid var(--line); border-radius:20px; padding:1.35rem 1.4rem; }
    .feature-number,.result-label { color:var(--blue); font-size:.75rem; font-weight:700; letter-spacing:.1em; text-transform:uppercase; }
    .feature-card h3 { font-size:1.12rem; margin:1.2rem 0 .5rem; }
    .feature-card p { color:var(--muted); font-size:.9rem; line-height:1.55; margin:0; }
    .upload-card { background:linear-gradient(135deg,#fff,#f4fcf8); border-color:#d5e5dc; }
    .upload-title,.result-value { font-family:'Space Grotesk'; font-size:1.45rem; font-weight:600; }
    .upload-copy { color:var(--muted); font-size:.92rem; line-height:1.5; margin:.45rem 0 1.2rem; }
    [data-testid="stFileUploader"] { background:white; border:1px dashed #a8bcb3; border-radius:14px; padding:.4rem; }
    [data-testid="stFileUploader"] button { background:var(--ink); border:0; border-radius:8px; color:white; }
    .status-row { color:#557064; font-size:.78rem; margin-top:1rem; }
    .status-dot { background:#28a66a; border-radius:50%; display:inline-block; height:8px; margin-right:.5rem; width:8px; }
    .result-value { color:var(--ink); margin:.35rem 0 0; }
    .disclaimer { background:#fff8e9; border:1px solid #f1dfb5; border-radius:14px; color:#765d2d; font-size:.82rem; line-height:1.55; margin-top:2rem; padding:.85rem 1rem; }
    .footer { border-top:1px solid var(--line); color:#8490a0; font-size:.78rem; margin-top:4rem; padding-top:1.2rem; }
    @media (max-width:640px) { .block-container { padding:3.75rem 1rem 3rem; } .topbar-note { display:none; } .hero { border-radius:20px; } .hero h1 { font-size:3.1rem; } }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_service() -> ModelService:
    return get_model_service()


def image_from_upload(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile) -> Image.Image:
    image = Image.open(io.BytesIO(uploaded_file.getvalue()))
    image.load()
    return image.convert("RGB")


def render_explanation(service: ModelService, image: Image.Image) -> None:
    explanation = service.explain(image)
    heatmap = Image.open(io.BytesIO(base64.b64decode(explanation["heatmap_base64"])))
    st.markdown('<div class="result-card"><div class="result-label">Interpretability layer</div><div class="result-value">Grad-CAM explanation</div></div>', unsafe_allow_html=True)
    st.image(heatmap, caption="Highlighted regions contributed to the predicted class score.", use_container_width=True)
    st.caption("Grad-CAM is an attribution aid for research inspection, not clinical evidence.")


def main() -> None:
    st.markdown(
        """
        <div class="topbar"><div class="brand"><div class="brand-mark">✦</div><div class="brand-name">Med<span>Flow</span> · XAI</div></div><div class="topbar-note">Research intelligence for pathology</div></div>
        <section class="hero"><div class="eyebrow">● &nbsp; Explainable histopathology</div><h1>See the signal<br><em>behind the image.</em></h1><p class="hero-copy">A focused research workspace for exploring PathMNIST-style tissue patches with transparent predictions and visual explanations.</p><div class="hero-pills"><div class="hero-pill">ResNet18 model</div><div class="hero-pill">9 tissue classes</div><div class="hero-pill">Grad-CAM enabled</div></div></section>
        <div class="section-kicker">A clearer workflow</div><h2 class="section-title">From pixels to perspective.</h2><p class="section-copy">MedFlow-XAI turns a small tissue image into a readable research signal — without hiding the uncertainty or the evidence.</p>
        """,
        unsafe_allow_html=True,
    )
    columns = st.columns(3, gap="medium")
    for column, (number, title, copy) in zip(columns, [
        ("01", "Upload simply", "Drop a PNG, JPG, or BMP patch into the workspace. Inputs are normalized to 28 × 28."),
        ("02", "Read the signal", "Review the predicted tissue class, confidence, and full probability distribution."),
        ("03", "Inspect the why", "Use a Grad-CAM overlay to see which regions contributed to the prediction."),
    ]):
        with column:
            st.markdown(f'<div class="feature-card"><div class="feature-number">{number}</div><h3>{title}</h3><p>{copy}</p></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-kicker">Try the workspace</div><h2 class="section-title">Bring an image into focus.</h2>', unsafe_allow_html=True)
    upload_column, context_column = st.columns([1.25, .75], gap="large")
    with upload_column:
        st.markdown('<div class="upload-card"><div class="upload-title">Start an analysis</div><div class="upload-copy">Upload one tissue patch to generate a prediction and visual attribution map.</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "bmp"], help="Images are converted to RGB and resized to 28 x 28 pixels.", label_visibility="collapsed")
        st.markdown('<div class="status-row"><span class="status-dot"></span>Cloud inference · images processed in your app session</div></div>', unsafe_allow_html=True)
    with context_column:
        st.markdown('<div class="result-card"><div class="result-label">Model context</div><div class="result-value">ResNet18</div><p class="section-copy">Trained on PathMNIST-style histopathology patches across nine tissue classes.</p><p><strong>28 × 28 px</strong> input &nbsp;·&nbsp; <strong>9 classes</strong> output</p></div>', unsafe_allow_html=True)
    if uploaded_file is None:
        st.markdown('<div class="disclaimer"><strong>Research use only.</strong> MedFlow-XAI is an educational classifier and attribution aid — not a clinical diagnostic tool.</div><div class="footer">MedFlow-XAI · Built for transparent model research</div>', unsafe_allow_html=True)
        return

    image = image_from_upload(uploaded_file)
    st.markdown('<div class="section-kicker">Analysis complete</div><h2 class="section-title">Your tissue signal.</h2>', unsafe_allow_html=True)
    st.image(image, caption="Uploaded tissue patch", width=280)
    try:
        service = load_service()
        prediction = service.predict(image)
        st.markdown(f'<div class="result-card"><div class="result-label">Predicted tissue class</div><div class="result-value">{prediction["predicted_label"]}</div><div class="status-row"><span class="status-dot"></span>{prediction["confidence"]:.1%} model confidence · {prediction["device"]} inference</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-kicker">Distribution</div><h2 class="section-title">The full picture.</h2>', unsafe_allow_html=True)
        probabilities = pd.DataFrame({"Class": list(prediction["probabilities"]), "Probability": list(prediction["probabilities"].values())}).set_index("Class")
        st.bar_chart(probabilities)
        with st.spinner("Generating Grad-CAM explanation..."):
            render_explanation(service, image)
        st.markdown('<div class="disclaimer"><strong>Keep the context.</strong> Confidence is a model score, not a probability of disease. Grad-CAM highlights model attention, not clinical evidence.</div>', unsafe_allow_html=True)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        st.error(f"Model service unavailable: {exc}")


if __name__ == "__main__":
    main()
