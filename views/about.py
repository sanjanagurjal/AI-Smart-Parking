import streamlit as st

from common import hero

hero("ABOUT THE PROJECT", "AI-Based Smart Parking Slot Detection",
     "A computer-vision project that identifies available and occupied parking spaces from uploaded images.",
     "Final-year BSc IT project")

st.markdown("## Project overview")
st.write(
    "ParkSight AI uses a trained TensorFlow convolutional neural network "
    "(CNN) to classify parking-space images. For full-lot analysis, an "
    "XML file provides the coordinates of individual spaces. The system "
    "crops each space, predicts its occupancy, and displays the results "
    "as a colour-coded overlay."
)

st.markdown("## Technologies used")
t1, t2, t3 = st.columns(3)
t1.markdown("**Python**\n\nApplication logic")
t2.markdown("**TensorFlow / Keras**\n\nTrained CNN model")
t3.markdown("**Streamlit + OpenCV**\n\nWeb interface and image processing")

st.markdown("## Workflow")
st.write(
    "Upload image and XML → extract parking-space regions → "
    "classify each crop → display results → download annotated image."
)
st.warning(
    "The current version works with uploaded images. "
    "Live camera feeds, real-time monitoring, and IoT hardware are not implemented."
)
