"""ParkSight AI - entry point. Run with:  streamlit run app.py
Needs Streamlit >= 1.46 for the top navigation bar:  pip install -U streamlit
"""
import streamlit as st

from common import apply_css, get_model

st.set_page_config(page_title="ParkSight AI | Parking Intelligence", page_icon="🅿️", layout="wide")
apply_css()

if "lot_results" not in st.session_state:
    st.session_state.lot_results = None

get_model()  # fail early with a clear message if the model file is missing

# Sidebar: branding + threshold (shared by every page, read via st.session_state.threshold)
st.sidebar.markdown("""
<div class="brand-lockup">
    <div class="brand-mark">P</div>
    <div>
        <p class="brand-name">ParkSight AI</p>
        <p class="brand-caption">Parking intelligence</p>
    </div>
</div>
""", unsafe_allow_html=True)
st.sidebar.slider(
    "Occupied threshold", 0.05, 0.95, 0.50, 0.05, key="threshold",
    help="A slot is occupied when its predicted probability reaches this value.",
)
st.sidebar.markdown(
    '<p style="color:#33495e;font-size:0.85rem;margin:0 0 0.5rem;">AI model: TensorFlow CNN</p>'
    '<p style="color:#33495e;font-size:0.85rem;margin:0;">Analysis: Uploaded images, not live camera footage.</p>',
    unsafe_allow_html=True,
)

# Top navigation bar
nav = st.navigation(
    [
        st.Page("views/home.py", title="Home", default=True),
        st.Page("views/dashboard.py", title="Dashboard"),
        st.Page("views/detect.py", title="Detect Parking"),
        st.Page("views/single_slot.py", title="Single Slot"),
        st.Page("views/about.py", title="About"),
    ],
    position="top",
)
nav.run()

st.markdown("---")
st.markdown(
    '<p style="color:#33495e;font-size:0.85rem;">ParkSight AI · AI-Based Smart Parking Slot Detection</p>',
    unsafe_allow_html=True,
)
