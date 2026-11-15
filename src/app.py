from __future__ import annotations

import streamlit as st

from src.ui import apply_theme, render_disclaimer, render_sidebar_brand

st.set_page_config(page_title="MedFlow-XAI", page_icon="✦", layout="wide")
apply_theme()


def render_home() -> None:
    apply_theme()
    st.markdown(
        """
        <section class="hero">
          <div class="eyebrow">● &nbsp; Explainable histopathology</div>
          <h1>See the signal<br><em>behind the image.</em></h1>
          <p class="hero-copy">MedFlow-XAI is a multi-page research workspace for PathMNIST-style tissue classification. Explore predictions, visual explanations, measured evaluation, and model comparisons without one long scrolling page.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("## A focused research workspace")
    st.write("Select a page from the sidebar. Analyze is the upload workspace.")
    columns = st.columns(4)
    for column, (title, copy, icon, path) in zip(
        columns,
        [
            ("Analyze", "Upload a tissue image and inspect prediction confidence with Grad-CAM.", "↗", "/Analyze"),
            ("Evaluation", "Review accuracy, precision, recall, F1-score, and the confusion matrix.", "▣", "/Evaluation"),
            ("Comparison", "Compare baseline CNN and ResNet18 experiments side by side.", "⇄", "/Comparison"),
            ("Research", "Explore the architecture, reproducibility notes, and study limitations.", "⌁", "/Research"),
        ],
    ):
        with column:
            st.markdown(
                f'<a class="card card-link" href="{path}" target="_self"><span class="card-title">{icon} {title}</span><span class="muted card-copy">{copy}</span></a>',
                unsafe_allow_html=True,
            )
    st.markdown('<div class="workspace-card-gap"></div>', unsafe_allow_html=True)
    st.info("Use the sidebar to continue. Open Analyze to upload an image or try a demo sample.")
    render_disclaimer()
    st.markdown('<div class="footer">MedFlow-XAI · Built for transparent model research</div>', unsafe_allow_html=True)


pages = [
    st.Page(render_home, title="Home", default=True),
    st.Page("pages/1_Analyze.py", title="Analyze / Upload"),
    st.Page("pages/2_Evaluation.py", title="Evaluation"),
    st.Page("pages/3_Comparison.py", title="CNN vs ResNet18"),
    st.Page("pages/4_Research.py", title="Research"),
    st.Page("pages/5_History.py", title="Prediction history"),
]

render_sidebar_brand()
navigation = st.navigation(pages, position="sidebar")
navigation.run()
