import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load Model
MODEL_PATH = "models/plant_disease_model.keras"
model = load_model(MODEL_PATH)

# Load Class Labels
with open("models/class_labels.txt", "r") as f:
    CLASS_NAMES = [line.strip() for line in f.readlines()]

# Streamlit UI
st.set_page_config(page_title="🌱 Plant Disease Detection", layout="centered")

st.title("🌿 Plant Disease Detection")
st.write("Upload a plant leaf image and get disease prediction")

# Upload Image
uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess Image
    img = img.resize((128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = round(100 * np.max(prediction), 2)

    # Clean folder-style names
    predicted_class = predicted_class.replace("___", " ").replace("_", " ")

    # Split into plant + disease
    if " " in predicted_class:
        parts = predicted_class.split(" ", 1)
        plant = parts[0]
        disease = parts[1]
    else:
        plant = predicted_class
        disease = "Healthy"

    # Show clean output
    st.success(f"🌱 Plant: {plant}\n🦠 Disease: {disease}\n📊 Confidence: {confidence}%")
