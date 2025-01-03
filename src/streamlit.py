import os
import mlflow
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the RUN_ID from environment variables
model_uri = model_uri = f"runs:/{os.getenv('RUN_ID')}/model"  # Replace the RUN_ID with the your MLflow run ID in your .env file

# Load the trained model
model = mlflow.tensorflow.load_model(model_uri)

# Define image size
IMG_SIZE = (224, 224)

# Class labels (adjust according to your dataset)
class_labels = [
    "Ayam Betutu",
    "Beberuk Terong",
    "Coto Makasar",
    "Gudeg",
    "Kerak Telur",
    "Mie Aceh",
    "Nasi Kuning",
    "Nasi Pecel",
    "Papeda",
    "Pempek",
    "Peuyeum",
    "Rawon",
    "Rendang",
    "Sate Madura",
    "Serabi",
    "Soto Banjar",
    "Soto Lamongan",
    "Tahu SUmedang",
]

st.title("Indonesian Food Classifier")

st.write("Upload an image of Indonesian food, and the model will classify it!")

uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = load_img(uploaded_file, target_size=IMG_SIZE)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = image_array / 255.0

    predictions = model.predict(image_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)

    st.write(f"Predicted Class: {class_labels[predicted_class]}")
    st.write(f"Confidence: {confidence:.2f}")
