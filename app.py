import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
MODEL_PATH = "best_model.h5"        # must be in same repo
IMG_SIZE = (224, 224)
CLASS_NAMES = ["bird", "drone"]     # update if reversed in training

# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# ---------------------------------------------------------
# PREPROCESS FUNCTION
# ---------------------------------------------------------
def preprocess_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    # Resize to CNN input size
    img_resized = cv2.resize(image_np, IMG_SIZE)

    # Normalize
    img_resized = img_resized.astype("float32") / 255.0
    img_resized = np.expand_dims(img_resized, axis=0)

    return image, img_resized

# ---------------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------------
st.title("🐦🚁 Bird vs Drone Classifier")
st.write("Upload an image to classify it as **Bird** or **Drone**.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    image, processed = preprocess_image(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Predict
    pred_prob = model.predict(processed)[0][0]

    # Determine class
    if pred_prob > 0.5:
        predicted_class = CLASS_NAMES[1]   # drone
        confidence = pred_prob
    else:
        predicted_class = CLASS_NAMES[0]   # bird
        confidence = 1 - pred_prob

    # Display Output
    st.subheader("Prediction")
    st.write(f"**Class:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.4f}")
