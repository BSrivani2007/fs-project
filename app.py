import os
import requests
import tensorflow as tf
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

MODEL_URL = "https://github.com/BSrivani2007/fs-project/releases/download/v1.0/fruit_freshness_model.keras"
MODEL_PATH = "fruit_freshness_model.keras"

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading AI model...")
        r = requests.get(MODEL_URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(r.content)
        print("Model downloaded!")

download_model()
model = tf.keras.models.load_model(MODEL_PATH)


st.title("🍎 Fruit & Vegetable Freshness Detector")

uploaded_file = st.file_uploader("Upload a fruit or vegetable image")

if uploaded_file:
    img = Image.open(uploaded_file).resize((224,224))
    img = np.array(img)/255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    class_names = ['FreshApple','FreshMango','FreshOrange','FreshPotato','FreshTomato',
                   'RottenApple','RottenMango','RottenOrange','RottenPotato','RottenTomato']

    result = class_names[np.argmax(pred)]

    st.success(f"Prediction: {result}")

