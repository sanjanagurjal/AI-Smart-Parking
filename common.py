"""Shared code: model, helpers, CSS. Logic copied unchanged from the original app.py."""
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf

CROP = 96
MODEL_PATH = "parking_model.keras"
GREEN, RED = (0, 200, 0), (255, 0, 0)  # RGB

def apply_css():
    css_path = Path(__file__).with_name("style.css")
    css = css_path.read_text(encoding="utf-8") if css_path.exists() else ""
    if len(css) < 500:
        st.sidebar.warning("style.css is missing or empty. Replace it with the new file.")

    # Always-on fixes for faint text (sidebar and footer)
    extra = """
    [data-testid="stSidebar"], [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div { color: #203950 !important; }
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p,
    [data-testid="stSidebar"] [data-testid="stSliderTickBarMin"],
    [data-testid="stSidebar"] [data-testid="stSliderTickBarMax"] { color: #526b81 !important; }
    [data-testid="stSidebar"] [data-testid="stThumbValue"] { color: #1768c4 !important; }
    [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color: #526b81 !important; }
    """
    st.html(f"<style>{css}\n{extra}</style>")


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def get_model():
    try:
        return load_model()
    except Exception as e:
        st.error(f"Could not load `{MODEL_PATH}`. Keep the model file next to app.py.\n\n{e}")
        st.stop()


def hero(kicker, title, subtitle, badge):
    st.markdown(f"""
    <div class="dashboard-hero">
        <div>
            <p class="dashboard-kicker">{kicker}</p>
            <h1 class="dashboard-title">{title}</h1>
            <p class="dashboard-subtitle">{subtitle}</p>
        </div>
        <div class="hero-model">{badge}</div>
    </div>
    """, unsafe_allow_html=True)


def order_points(pts):
    pts = np.asarray(pts, dtype="float32")
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], dtype="float32")


def parse_spaces(xml_bytes):
    """Read a PKLot XML file. Returns list of {id, occupied (1/0/None), pts (4x2)}."""
    root = ET.fromstring(xml_bytes)
    spaces = []
    for sp in root.iter("space"):
        pts = [(float(p.get("x")), float(p.get("y"))) for p in sp.iter("point")]
        if len(pts) != 4:
            rr = sp.find("rotatedRect")
            if rr is not None:
                c, sz, a = rr.find("center"), rr.find("size"), rr.find("angle")
                box = cv2.boxPoints(((float(c.get("x")), float(c.get("y"))),
                                     (float(sz.get("w")), float(sz.get("h"))),
                                     float(a.get("d"))))
                pts = [tuple(p) for p in box]
            elif len(pts) >= 3:
                pts = [tuple(p) for p in cv2.boxPoints(cv2.minAreaRect(np.array(pts, dtype="float32")))]
            else:
                continue
        occ = sp.get("occupied")
        spaces.append({"id": sp.get("id"),
                       "occupied": int(occ) if occ not in (None, "") else None,
                       "pts": order_points(pts)})
    return spaces


def crop_space(img, pts, size=CROP):
    dst = np.array([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(np.asarray(pts, dtype="float32"), dst)
    return cv2.warpPerspective(img, M, (size, size))


def draw_overlay(img_rgb, spaces, occupied_flags):
    out, layer = img_rgb.copy(), img_rgb.copy()
    for s, occ in zip(spaces, occupied_flags):
        color = RED if occ else GREEN
        pts = np.round(s["pts"]).astype(np.int32)
        cv2.fillPoly(layer, [pts], color)
        cv2.polylines(out, [pts], True, color, 2)
    return cv2.addWeighted(layer, 0.35, out, 0.65, 0)
