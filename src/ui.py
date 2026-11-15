from __future__ import annotations

import base64
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import streamlit as st
from PIL import Image

from src.api.main import ModelService, get_model_service

ROOT_DIR = Path(__file__).resolve().parents[1]
FIGURES_DIR = ROOT_DIR / "paper" / "figures"
RESULTS_DIR = ROOT_DIR / "results"


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
        :root { --ink:#0b1220; --muted:#647084; --line:#e5e9ee; --mint:#c7f7df; --blue:#4d6fff; }
        html, body { overflow-x:hidden; }
        .stApp { background:#f6f8fa; color:var(--ink); }
        .block-container { width:100%; max-width:1180px; padding:2.5rem 2.5rem 4rem; }
        header[data-testid="stHeader"] { background:transparent !important; border-bottom:0 !important; }
        h1,h2,h3,p,div,button,label { font-family:'DM Sans',sans-serif; }
        h1,h2,h3 { font-family:'Space Grotesk',sans-serif; letter-spacing:-.04em; }
        h1, h2, h3, p { overflow-wrap:anywhere; }
        [data-testid="stSidebar"] { background:var(--ink); }
        [data-testid="stSidebar"] * { color:#e7edf5; }
        [data-testid="stSidebarHeader"] { position:relative !important; z-index:20 !important; }
        .brand { position:fixed !important; top:1rem; left:1.5rem; display:grid; grid-template-columns:auto 1fr; align-items:center; column-gap:.7rem; padding:0; pointer-events:none !important; z-index:1 !important; }
        .brand-mark { display:flex; align-items:center; justify-content:center; width:38px; height:38px; border-radius:12px; background:var(--mint); color:var(--ink); font-weight:700; }
        .brand-name { font-family:'Space Grotesk'; font-weight:700; font-size:1.1rem; }
        .brand-name span { color:var(--blue); }
        .brand-subtitle { grid-column:2; color:#9eacbd !important; font-size:.75rem; line-height:1.3; margin-top:-.25rem; }
        [data-testid="stSidebarNav"] { padding-top:5rem; }
        .eyebrow,.section-kicker { color:var(--blue); font-size:.72rem; font-weight:700; letter-spacing:.15em; text-transform:uppercase; }
        .hero { background:var(--ink); border-radius:26px; color:white; padding:3.3rem 4rem; margin-bottom:2rem; }
        .hero h1 { font-size:clamp(2.4rem,6vw,5.3rem); line-height:.95; margin:1rem 0; }
        .hero h1 em { color:var(--mint); font-style:normal; }
        .hero-copy,.muted { color:var(--muted); line-height:1.65; }
        .hero-copy { color:#b8c3d0; max-width:650px; font-size:1.05rem; }
        .card { display:flex; flex-direction:column; background:white; border:1px solid var(--line); border-radius:18px; box-sizing:border-box; height:180px; min-height:180px; padding:1.25rem 1.25rem 1.5rem !important; }
        .card-copy { min-height:0; line-height:1.5 !important; }
        .card-title { display:block; color:var(--ink); font-family:'Space Grotesk',sans-serif; font-size:1.35rem; font-weight:600; letter-spacing:-.03em; line-height:1.2; margin-bottom:.9rem; }
        .card h3 { margin-top:0; margin-bottom:.9rem; }
        .card p { margin:0; }
        .card-bottom-space { height:.75rem; }
        .workspace-card-gap { height:1.5rem; }
        .card-link { color:var(--ink); text-decoration:none !important; }
        .card-link:hover { color:var(--blue); }
        .metric-card { background:white; border:1px solid var(--line); border-radius:16px; padding:1rem; }
        .metric-label { color:var(--muted); font-size:.75rem; text-transform:uppercase; letter-spacing:.08em; }
        .metric-value { font-family:'Space Grotesk'; font-size:1.7rem; font-weight:600; margin-top:.3rem; }
        .disclaimer { background:#fff8e9; border:1px solid #f1dfb5; border-radius:14px; color:#765d2d; font-size:.84rem; line-height:1.55; padding:.9rem 1rem; margin-top:1.4rem; }
        .footer { border-top:1px solid var(--line); color:#8490a0; font-size:.78rem; margin-top:3rem; padding-top:1.2rem; }
        [data-testid="stDataFrame"] { max-width:100%; overflow-x:auto; }
        [data-testid="stImage"] img { max-width:100%; height:auto; }
        div.stButton > button, div.stDownloadButton > button { min-height:2.75rem; }
        @media (max-width: 640px) {
            .block-container { padding:1.25rem 1rem 2.5rem; }
            [data-testid="stSidebar"][aria-expanded="true"] {
                min-width:min(86vw, 20rem) !important;
                max-width:86vw !important;
            }
            .brand { top:1rem; left:1rem; }
            [data-testid="stHorizontalBlock"] {
                flex-direction:column !important;
                gap:1rem !important;
            }
            [data-testid="stHorizontalBlock"] > [data-testid="column"] {
                width:100% !important;
                flex:1 1 100% !important;
                min-width:100% !important;
            }
            .hero {
                border-radius:18px;
                padding:2rem 1.25rem;
                margin-bottom:1.5rem;
            }
            .hero h1 {
                font-size:clamp(2.35rem, 12vw, 3.5rem);
                line-height:.98;
                margin:.85rem 0;
            }
            .hero-copy { font-size:.98rem; }
            .card { height:180px; min-height:180px; padding:1rem !important; }
            .card-title { font-size:1.2rem; margin-bottom:.65rem; }
            .workspace-card-gap { height:.25rem; }
            .metric-card { padding:.85rem .9rem; }
            .metric-value {
                font-size:1.3rem;
                line-height:1.2;
                overflow-wrap:anywhere;
            }
            .disclaimer { margin-top:1rem; }
            .footer { margin-top:2rem; }
            div.stButton > button, div.stDownloadButton > button {
                width:100%;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def render_top_brand() -> None:
    st.markdown(
        '<div class="top-brand"><div class="brand-mark">✦</div><div class="brand-name">Med<span>Flow</span> · XAI</div></div>',
        unsafe_allow_html=True,
    )

def render_sidebar_brand() -> None:
    st.sidebar.markdown(
        '<div class="brand"><div class="brand-mark">✦</div><div class="brand-name">Med<span>Flow</span> · XAI</div><div class="brand-subtitle">Research workspace · PathMNIST</div></div>',
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_service() -> ModelService:
    return get_model_service()


def image_from_bytes(contents: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(contents))
    image.load()
    return image.convert("RGB")


def image_from_upload(uploaded_file: Any) -> Image.Image:
    return image_from_bytes(uploaded_file.getvalue())


def sample_images() -> list[tuple[str, Path]]:
    return [
        ("Correct example 01", FIGURES_DIR / "correct_01.png"),
        ("Correct example 02", FIGURES_DIR / "correct_02.png"),
        ("Wrong example 01", FIGURES_DIR / "wrong_01.png"),
        ("Wrong example 02", FIGURES_DIR / "wrong_02.png"),
    ]


def explanation_image(explanation: dict[str, object]) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(str(explanation["heatmap_base64"]))))


def add_history(prediction: dict[str, object], source: str) -> None:
    history = st.session_state.setdefault("prediction_history", [])
    if history:
        latest = history[0]
        if (
            latest["source"] == source
            and latest["prediction"] == prediction["predicted_label"]
            and latest["confidence"] == float(prediction["confidence"])
        ):
            return
    history.insert(
        0,
        {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source": source,
            "prediction": prediction["predicted_label"],
            "confidence": float(prediction["confidence"]),
            "device": prediction["device"],
        },
    )
    del history[25:]


def report_bytes(prediction: dict[str, object]) -> bytes:
    rows = [
        "MedFlow-XAI prediction report",
        f"Generated (UTC),{datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"Predicted class,{prediction['predicted_label']}",
        f"Confidence,{float(prediction['confidence']):.6f}",
        f"Device,{prediction['device']}",
        "",
        "Class,Probability",
    ]
    rows.extend(
        f"{label},{float(probability):.6f}"
        for label, probability in dict(prediction["probabilities"]).items()
    )
    return ("\n".join(rows) + "\n").encode("utf-8")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def render_disclaimer() -> None:
    st.markdown(
        '<div class="disclaimer"><strong>Research use only.</strong> MedFlow-XAI is an educational classifier and attribution aid, not a clinical diagnostic tool. Model confidence and Grad-CAM highlights are not clinical evidence.</div>',
        unsafe_allow_html=True,
    )
