"""
PlantXAI-Net — Streamlit interface
Reuses the existing app.py model and five-stage inference pipeline.
"""
import os

# Set thread limits before PyTorch is imported through app.py.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import streamlit as st
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="PlantXAI-Net",
    page_icon="🌱",
    layout="wide",
)

@st.cache_resource(show_spinner="Loading PlantXAI-Net EfficientNet-B0 model...")
def load_pipeline():
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    import app as plantxai
    return plantxai

st.markdown(
    """
    <div style="background:linear-gradient(135deg,#1e3a5f,#2d6a4f);
                color:white;border-radius:12px;padding:22px 26px;margin-bottom:18px;">
      <h1 style="margin:0;color:white;">🌱 PlantXAI-Net</h1>
      <p style="margin:8px 0 0 0;">
        Explainable AI for Plant Disease Detection &amp; Recommendation
        · EfficientNet-B0 · Grad-CAM XAI · 38 Classes
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    "**Five-stage pipeline:** 1. Preprocessing → 2. Disease classification → "
    "3. Grad-CAM explanation → 4. Severity estimation → 5. Treatment recommendation"
)

with st.sidebar:
    st.header("📤 Upload plant image")
    uploaded_file = st.file_uploader(
        "Choose a leaf image",
        type=["jpg", "jpeg", "png", "webp"],
        help="Use a clear, well-lit image focused on a single leaf.",
    )
    st.caption("Supported formats: JPG, JPEG, PNG and WEBP.")
    st.info("For best results, use a clear, well-lit photo of a single leaf.")

if uploaded_file is None:
    st.info("Upload a plant leaf image from the sidebar to begin analysis.")
    st.stop()

try:
    image = Image.open(uploaded_file).convert("RGB")
except Exception as exc:
    st.error(f"Could not read the uploaded image: {exc}")
    st.stop()

col1, col2 = st.columns(2)
with col1:
    st.subheader("Uploaded image")
    st.image(image, use_container_width=True)

if st.button("🔍 Analyze plant image", type="primary", use_container_width=True):
    try:
        pipeline = load_pipeline()
        image_array = np.asarray(image, dtype=np.uint8)
        with st.spinner("Running the five-stage PlantXAI-Net pipeline..."):
            preprocessed, gradcam_overlay, stage1, stage2, stages345, full_html = (
                pipeline.run_pipeline(image_array)
            )
        st.session_state["plantxai_result"] = {
            "preprocessed": preprocessed,
            "gradcam": gradcam_overlay,
            "stage1": stage1,
            "stage2": stage2,
            "stages345": stages345,
            "full_html": full_html,
        }
    except Exception as exc:
        st.error(f"Analysis failed: {exc}")
        st.exception(exc)

result = st.session_state.get("plantxai_result")
if result is not None:
    st.divider()
    st.subheader("📊 Full pipeline results")
    st.markdown(result["full_html"], unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(
        ["🖼️ Preprocessed image", "🔥 Grad-CAM heatmap", "📋 Stage details"]
    )
    with tab1:
        st.image(
            result["preprocessed"],
            caption="224 × 224 preprocessed image",
            use_container_width=True,
        )
    with tab2:
        st.image(
            result["gradcam"],
            caption="Grad-CAM overlay — highlighted regions indicate model attention",
            use_container_width=True,
        )
    with tab3:
        st.markdown("### Stage 1 — Image preprocessing")
        st.markdown(result["stage1"], unsafe_allow_html=True)
        st.markdown("### Stage 2 — Disease classification")
        st.markdown(result["stage2"], unsafe_allow_html=True)
        st.markdown("### Stages 3–5 — XAI, severity and recommendation")
        st.markdown(result["stages345"], unsafe_allow_html=True)

st.divider()
st.caption(
    "PlantXAI-Net · EfficientNet-B0 · Grad-CAM · "
    "For informational purposes only. Consult a qualified agronomist "
    "before making crop-treatment decisions."
)
