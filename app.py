import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("fruit_freshness_model.keras")

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
