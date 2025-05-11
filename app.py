import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(page_title="♻️ Garbage Classification App", layout="wide")

# ---------------------- SIDEBAR ----------------------
theme_choice = st.sidebar.selectbox(
    "Select Background Theme", ["Green", "Dark"])
view_mode = st.sidebar.radio(
    "Choose layout mode", ("Portrait (scrolling)", "Landscape (side by side)"))

# ---------------------- STYLING FUNCTION ----------------------


def set_theme(theme):
    if theme == "Green":
        background_color = "#a3d9a5"
        text_color = "white"
        extra_styles = """
        [data-testid="baseButton-secondary"] {
            background-color: white !important;
            color: white !important;
            border: 1px solid white !important;
        }
        .st-emotion-cache-1j4k5q1 {
            background-color: white !important;
            color: white !important;
            border: 1px dashed white !important;
        }
        .st-emotion-cache-1vbkxwb {
            background-color: white !important;
            color: white !important;
        }

        div[data-baseweb="select"] * {
            color: white !important;
        }
        div[data-testid="stFileUploader"] label {
            color: white !important;
        }
        """
    else:
        background_color = "#000000"
        text_color = "white"
        extra_styles = ""

    css = f"""
    <style>
    html, body, [data-testid="stAppViewContainer"] {{
        background-color: {background_color} !important;
        color: {text_color} !important;
        transition: background-color 0.3s ease;
    }}
    h1, h2, h3, h4, h5, h6, p, span, label, div, section {{
        color: {text_color} !important;
    }}
    img {{
        max-width: 100%;
        height: auto;
    }}
    @media (max-width: 768px) {{
        .stImage {{
            display: block;
            margin: 0 auto;
        }}
    }}
    {extra_styles}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# ---------------------- APPLY THEME ----------------------
set_theme(theme_choice)

# ---------------------- HEADER ----------------------
st.markdown("<h1 style='text-align: center; font-size: 50px;'>♻️ Garbage Classification App</h1>",
            unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; font-size: 20px;'>Upload a waste image to predict its category</h2>", unsafe_allow_html=True)


# ---------------------- MODEL LOADING ----------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model_epoch_55.h5")


model = load_model()
class_labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# ---------------------- FILE UPLOADER ----------------------
uploaded_file = st.file_uploader(
    "Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with st.spinner("🔍 Predicting..."):
        image = Image.open(uploaded_file)

        # Resize and normalize
        img = image.resize((180, 180))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        predicted_label = class_labels[np.argmax(prediction)]
        confidence_scores = prediction[0]

    # ---------------------- DISPLAY SECTION ----------------------
    if view_mode == "Portrait (scrolling)":
        st.markdown("<div style='text-align: center;'>",
                    unsafe_allow_html=True)
        st.image(image, caption="Uploaded Image",
                 use_column_width=False, width=300)
        st.markdown(
            f"<h3>🧠 Predicted Class: <strong>{predicted_label.upper()}</strong></h3>", unsafe_allow_html=True)

        st.markdown("### 🔍 Prediction Confidence")
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.barh(class_labels, confidence_scores, color='teal')
        ax.set_xlabel("Confidence")
        ax.set_xlim([0, 1])
        ax.invert_yaxis()
        for i, v in enumerate(confidence_scores):
            ax.text(v + 0.01, i, f"{v:.2f}", va='center')
        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)

    else:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(image, caption="Uploaded Image",
                     use_column_width=False, width=300)
            st.markdown(
                f"<h3>🧠 Predicted Class: <strong>{predicted_label.upper()}</strong></h3>", unsafe_allow_html=True)
        with col2:
            st.markdown("### 🔍 Prediction Confidence")
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.barh(class_labels, confidence_scores, color='slateblue')
            ax.set_xlabel("Confidence")
            ax.set_xlim([0, 1])
            ax.invert_yaxis()
            for i, v in enumerate(confidence_scores):
                ax.text(v + 0.01, i, f"{v:.2f}", va='center')
            st.pyplot(fig)
