from __future__ import annotations

import pandas as pd
import streamlit as st

from src.ui import apply_theme, render_disclaimer

st.set_page_config(page_title="History · MedFlow-XAI", page_icon="✦", layout="wide")
apply_theme()
st.title("Prediction history")
st.write("Predictions are kept for this browser session only. No image bytes are stored in the history.")

history = st.session_state.get("prediction_history", [])
if not history:
    st.info("No predictions yet. Visit Analyze to run a prediction.")
else:
    frame = pd.DataFrame(history)
    frame["confidence"] = frame["confidence"].map(lambda value: f"{value:.1%}")
    st.dataframe(frame, use_container_width=True, hide_index=True)
    st.download_button(
        "Download history (CSV)",
        data=frame.to_csv(index=False).encode("utf-8"),
        file_name="medflow_prediction_history.csv",
        mime="text/csv",
    )
    if st.button("Clear session history"):
        st.session_state["prediction_history"] = []
        st.rerun()

render_disclaimer()
