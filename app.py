import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import pandas as pd
from datetime import datetime

# Create directory for saving user images
os.makedirs("user_images", exist_ok=True)

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(page_title="♻️ Garbage Classification App", layout="wide")

# ---------------------- SIDEBAR ----------------------
# Navigation
page = st.sidebar.radio("📄 Navigate", ["Home", "About"])

# Theme and View Mode
theme_choice = st.sidebar.selectbox(
    "🎨 Select Background Theme", ["Dark", "Green"])
view_mode = st.sidebar.radio(
    "🖼️ Choose layout mode", ("Portrait (scrolling)", "Landscape (side by side)"))

# ---------------------- THEME FUNCTION ----------------------


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

# ---------------------- MODEL LOADING ----------------------


@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model_epoch_55.h5")


model = load_model()
class_labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# ---------------------- FEEDBACK SAVE FUNCTION ----------------------


def save_feedback(image, predicted_label, feedback, correct_label=None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_filename = f"user_images/{timestamp}.png"
    image.save(image_filename)

    feedback_data = {
        "timestamp": timestamp,
        "predicted_class": predicted_label,
        "feedback": feedback,
        "correct_class": correct_label if feedback == "no" else predicted_label,
        "image_path": image_filename
    }

    feedback_file = "feedback_log.csv"
    if os.path.exists(feedback_file):
        df = pd.read_csv(feedback_file)
        df = pd.concat([df, pd.DataFrame([feedback_data])], ignore_index=True)
    else:
        df = pd.DataFrame([feedback_data])

    df.to_csv(feedback_file, index=False)


# ---------------------- SESSION STATE INIT ----------------------
if "file_uploader_key" not in st.session_state:
    st.session_state["file_uploader_key"] = 0

# ---------------------- PAGE HANDLER ----------------------
if page == "Home":
    # ---------------------- HEADER ----------------------
    st.markdown("<h1 style='text-align: center; font-size: 50px;'>♻️ Garbage Classification App</h1>",
                unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; font-size: 20px;'>Upload a waste image to predict its category</h2>", unsafe_allow_html=True)

    # ---------------------- FILE UPLOADER ----------------------
    uploaded_file = st.file_uploader("Choose an image...", type=[
                                     "jpg", "jpeg", "png"], key=st.session_state["file_uploader_key"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img_resized = image.resize((180, 180))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        with st.spinner("🔍 Predicting..."):
            prediction = model.predict(img_array)
            predicted_label = class_labels[np.argmax(prediction)]
            confidence_scores = prediction[0]

        # ---------------------- DISPLAY ----------------------
        if view_mode == "Portrait (scrolling)":
            st.markdown("<div style='text-align: center;'>",
                        unsafe_allow_html=True)
            st.image(image, caption="Uploaded Image", use_container_width=True)
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
                         use_container_width=True)
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

        # ---------------------- FEEDBACK ----------------------
        st.markdown("### 🗳️ Was this prediction correct?")
        feedback = st.radio("Your feedback:", ["yes", "no"], horizontal=True)
        correct_label = None

        if st.button("✅ Submit Feedback"):
            save_feedback(image, predicted_label, feedback, correct_label)
            st.success("✅ Feedback submitted. Thank you!")
            st.session_state.feedback_submitted = True

        if st.session_state.get("feedback_submitted"):
            if st.button("🔁 Try Another Image"):
                st.session_state.feedback_submitted = False
                st.session_state.uploaded_file = None
                st.session_state["file_uploader_key"] += 1
                st.rerun()

elif page == "About":
    st.title("📘 About This App")
    st.markdown("""
    ### ♻️ Garbage Classification App

    This app allows you to upload images of garbage and predicts the category using a deep learning model.

    **Categories:**  
    - Cardboard  
    - Glass  
    - Metal  
    - Paper  
    - Plastic  
    - Trash  

    **Built With:**  
    - TensorFlow  
    - Streamlit  
    - PIL, Matplotlib, Pandas  

    **Author:** Samuel Adebayo 
    """)
