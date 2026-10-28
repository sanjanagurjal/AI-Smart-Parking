import cv2
import numpy as np
import streamlit as st
from PIL import Image

from common import crop_space, draw_overlay, get_model, hero, parse_spaces

model = get_model()
threshold = st.session_state.get("threshold", 0.50)

hero("AI ANALYSIS", "Detect Parking Spaces",
     "Upload the parking image and its corresponding XML slot map.",
     "CNN · Image-based detection")

c1, c2 = st.columns(2, gap="large")
with c1:
    st.markdown("### 01 · Parking-lot image")
    img_file = st.file_uploader("Upload JPG or PNG", type=["jpg", "jpeg", "png"], key="lot_img")
with c2:
    st.markdown("### 02 · Parking slot map")
    xml_file = st.file_uploader("Upload matching XML", type=["xml"], key="lot_xml")

if img_file and xml_file:
    try:
        img = np.array(Image.open(img_file).convert("RGB"))
        spaces = parse_spaces(xml_file.getvalue())

        if not spaces:
            st.error("No parking slots found in this XML file.")
        else:
            crops = np.stack([crop_space(img, s["pts"]) for s in spaces]).astype("float32")

            probs = model.predict(crops, verbose=0).ravel()
            flags = probs >= threshold
            out = draw_overlay(img, spaces, flags)

            available = int((~flags).sum())
            occupied = int(flags.sum())
            total = len(spaces)
            occupancy = 100 * occupied / total

            st.session_state.lot_results = {
                "total": total, "available": available, "occupied": occupied,
                "occupancy": occupancy, "image": out,
            }

            st.markdown("## Detection results")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total slots", total)
            m2.metric("Available", available)
            m3.metric("Occupied", occupied)
            m4.metric("Occupancy rate", f"{occupancy:.0f}%")

            st.markdown(
                '<div class="result-legend">'
                '<span class="legend-item"><span class="legend-dot" style="background:#18a66a"></span>Available</span>'
                '<span class="legend-item"><span class="legend-dot" style="background:#e1535d"></span>Occupied</span></div>',
                unsafe_allow_html=True,
            )

            st.image(out, caption="Green = available · Red = occupied", use_container_width=True)

            ok, png = cv2.imencode(".png", cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
            if ok:
                st.download_button("Download annotated result", png.tobytes(),
                                   file_name="parking_result.png", mime="image/png")

            labelled = [i for i, s in enumerate(spaces) if s["occupied"] is not None]
            if labelled:
                truth = np.array([spaces[i]["occupied"] == 1 for i in labelled])
                agreement = (flags[labelled] == truth).mean() * 100
                st.caption(f"Agreement with XML labels: {agreement:.1f}% "
                           f"({int((flags[labelled] == truth).sum())}/{len(labelled)} slots)")

    except Exception as e:
        st.error(f"Could not analyze these files. Check that the image and XML match. Details: {e}")
else:
    st.info("Upload both files to start detection. The XML should describe the parking spaces in the selected image.")
