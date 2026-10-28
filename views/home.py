import streamlit as st

st.markdown("""
<div style="
    padding:3.5rem 2.5rem;
    border-radius:18px;
    background:linear-gradient(120deg,#102b49,#1768c4);
    color:white;
    margin-bottom:1.5rem;">
    <p style="color:#b9dcff;font-weight:700;letter-spacing:2px;">
        AI-POWERED PARKING INTELLIGENCE
    </p>
    <h1 style="color:white;font-size:3rem;line-height:1.15;">
        Smarter Parking.<br>Powered by AI.
    </h1>
    <p style="color:#e0edff;font-size:1.1rem;max-width:650px;">
        Detect available and occupied parking spaces from parking-lot
        images using computer vision and a trained deep-learning model.
    </p>
</div>
""", unsafe_allow_html=True)

left, right = st.columns([1, 1], gap="large")
with left:
    if st.button("Start Parking Detection →", type="primary", use_container_width=True):
        st.switch_page("views/detect.py")
with right:
    if st.button("Explore the Project", use_container_width=True):
        st.switch_page("views/about.py")

st.markdown("## What ParkSight AI can do")
a, b, c = st.columns(3)
a.markdown("""
###  Slot Detection
Classify individual parking spaces as available or occupied.
""")
b.markdown("""
###  Visual Analytics
View slot counts, occupancy rate, and prediction summaries.
""")
c.markdown("""
###  Image Results
Review colour-coded parking spaces and download the result.
""")

st.markdown("---")
st.markdown("## How it works")
x, y, z = st.columns(3)
x.markdown("**01 · Upload**\n\nChoose a parking-lot image and its matching XML slot map.")
y.markdown("**02 · Analyze**\n\nThe trained CNN classifies each detected parking-space crop.")
z.markdown("**03 · Review**\n\nInspect the coloured overlay and download the annotated image.")

st.info("This project analyzes uploaded images. It does not use live CCTV or physical IoT sensors.")
