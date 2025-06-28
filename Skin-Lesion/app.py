import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import base64
from io import BytesIO

# -------------------- CSS -------------------- #
st.markdown("""
    <style>
    .title-box {
        background-color: #f5e0e9;
        padding: 1rem 2rem;
        border-radius: 10px;
        font-weight: bold;
        font-size: 2rem;
        color: #3a3a3a;
        box-shadow: 0 10px 8px rgba(0, 0, 0, 0.1);
        margin: 2rem auto 1rem auto;
        text-align: center;
        width: fit-content;
    }
    .instruction-text {
        font-size: 0.9rem;
        color: #333;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .preview-img {
        display: block;
        margin: 1.5rem auto 1rem auto;
        border-radius: 10px;
        width: 300px;
        height: auto;
    }
    .prediction {
        text-align: center;
        font-size: 1.8rem;
        color: #2e8b57;
        font-weight: bold;
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------- CLASS LABELS -------------------- #
class_names = [
    'Actinic Keratosis', 'Basal Cell Cancer', 'Benign Keratosis',
    'Dermatofibroma', 'Melanoma', 'Nevus', 'Vascular Cancer'
]

# -------------------- LOAD MODEL -------------------- #
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('Hybrid_ResNet50_Inception.h5')
model = load_model()

# -------------------- HEADER -------------------- #
st.markdown('<div class="title-box">🩺 Skin Lesion Classification</div>', unsafe_allow_html=True)
st.markdown('<p class="instruction-text">Upload a skin lesion image. The model will predict the class of the lesion.</p>', unsafe_allow_html=True)

# -------------------- FILE UPLOADER -------------------- #
file = st.file_uploader("Upload a skin lesion image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

# -------------------- PREDICTION FUNCTION -------------------- #
def import_and_predict(image_data, model):
    size = (224, 224)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image).astype(np.float32)

    # Preprocess input for both branches
    resnet_input = tf.keras.applications.resnet50.preprocess_input(img_array.copy())
    inception_input = tf.keras.applications.inception_v3.preprocess_input(img_array.copy())

    resnet_input = np.expand_dims(resnet_input, axis=0)
    inception_input = np.expand_dims(inception_input, axis=0)

    prediction = model.predict([resnet_input, inception_input])
    return prediction

# -------------------- BASE64 IMAGE ENCODER -------------------- #
def image_to_base64(img: Image.Image) -> str:
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# -------------------- DISPLAY -------------------- #
if file is not None:
    image = Image.open(file).convert('RGB')
    predictions = import_and_predict(image, model)
    score = tf.nn.softmax(predictions[0])
    predicted_class = class_names[np.argmax(score)]

    # Convert image to base64 and embed it directly
    img_b64 = image_to_base64(image)
    st.markdown(f'<img src="data:image/png;base64,{img_b64}" class="preview-img">', unsafe_allow_html=True)

    # Show prediction
    st.markdown(f'<div class="prediction">🧠 Prediction: <strong>{predicted_class}</strong></div>', unsafe_allow_html=True)
else:
    st.warning("⚠ Please upload an image file.")
