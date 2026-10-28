import streamlit as st

from common import hero

hero("OVERVIEW", "Parking Dashboard",
     "Review the latest parking-lot analysis performed in this session.",
     "TensorFlow CNN · Local inference")

result = st.session_state.get("lot_results")
if result is None:
    st.info("No parking-lot analysis yet. Open Detect Parking to upload an image and its matching XML file.")
    if st.button("Go to Detect Parking"):
        st.switch_page("views/detect.py")
else:
    st.metric("Total slots", result["total"])
    m1, m2, m3 = st.columns(3)
    m1.metric("Available", result["available"])
    m2.metric("Occupied", result["occupied"])
    m3.metric("Occupancy rate", f'{result["occupancy"]:.0f}%')
    st.image(result["image"], caption="Latest annotated parking-lot image", use_container_width=True)
    st.caption("These figures are from your latest analysis in this session.")
