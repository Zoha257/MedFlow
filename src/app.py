from __future__ import annotations

import base64
import io

import pandas as pd
import streamlit as st
from PIL import Image

from src.api.main import ModelService, get_model_service

st.set_page_config(page_title="MedFlow-XAI", page_icon="🔬", layout="wide")


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
    st.subheader("Grad-CAM explanation")
    st.image(
        heatmap,
        caption="Highlighted regions contributed to the predicted class score.",
        use_container_width=True,
    )
    st.caption(
        "Grad-CAM is an attribution aid for research inspection, not clinical evidence."
    )


def main() -> None:
    st.title("MedFlow-XAI")
    st.write(
        "Upload a PathMNIST-style histopathology patch to obtain a research "
        "prediction, confidence scores, and a Grad-CAM explanation."
    )
    st.warning("Research/educational classifier only — not a clinical diagnostic tool.")

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["png", "jpg", "jpeg", "bmp"],
        help="Images are converted to RGB and resized to 28 x 28 pixels.",
    )
    if uploaded_file is None:
        st.info("Choose an image to begin.")
        return

    image = image_from_upload(uploaded_file)
    st.image(image, caption="Uploaded image", width=280)
    try:
        service = load_service()
        prediction = service.predict(image)
        st.subheader("Prediction")
        st.success(
            f"{prediction['predicted_label']} "
            f"({prediction['confidence']:.1%} confidence)"
        )

        st.subheader("Class probabilities")
        probabilities = pd.DataFrame(
            {
                "Class": list(prediction["probabilities"]),
                "Probability": list(prediction["probabilities"].values()),
            }
        ).set_index("Class")
        st.bar_chart(probabilities)

        with st.spinner("Generating Grad-CAM explanation..."):
            render_explanation(service, image)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        st.error(f"Model service unavailable: {exc}")


if __name__ == "__main__":
    main()
