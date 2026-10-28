import cv2
import numpy as np
import streamlit as st
from PIL import Image

from common import CROP, get_model, hero

model = get_model()
threshold = st.session_state.get("threshold", 0.50)

hero("INDIVIDUAL ANALYSIS", "Single-Slot Analysis",
     "Classify one cropped parking-space image.", "CNN · 96 × 96 input")

slot_file = st.file_uploader("Upload a parking-slot image", type=["jpg", "jpeg", "png"], key="slot_img")

if slot_file:
    try:
        slot = np.array(Image.open(slot_file).convert("RGB"))
        x = cv2.resize(slot, (CROP, CROP)).astype("float32")[None]
        p = float(model.predict(x, verbose=0).ravel()[0])
        label = "OCCUPIED" if p >= threshold else "AVAILABLE"

        preview, summary = st.columns([1, 1], gap="large")
        with preview:
            st.image(slot, caption="Uploaded parking space", use_container_width=True)
        with summary:
            st.markdown("### Prediction")
            st.metric("Probability occupied", f"{p:.1%}")
            if label == "OCCUPIED":
                st.error("Occupied slot")
            else:
                st.success("Available slot")
            st.progress(min(max(p, 0.0), 1.0))
            st.caption(f"Probability of occupied: {p:.1%} · Threshold: {threshold:.0%}")

        if 0.35 < p < 0.65:
            st.warning("The model may be uncertain for this image.")
    except Exception as e:
        st.error(f"Could not process this image: {e}")
