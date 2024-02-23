import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import json

model = tf.keras.models.load_model("models/tumor_classifier.h5")

with open("tumor_classes.json", "r") as file:
    class_labels = json.load(file)

def classify_images(image):
    img = Image.open(image)
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    print(predictions)
    predicted_class = str(np.argmax(predictions[0]))
    confidence = int(np.max(predictions[0]))
    print(np.argmax(predictions[0]))
    print(predicted_class, confidence)
    return predicted_class, confidence


st.title('Tumor Image Classifier')

uploaded_file = st.file_uploader("Choose an Image..", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    st.write("")

    if st.button("classify"):
        st.write("Classifying")
        class_index, confidence = classify_images(uploaded_file)
        st.write(f"Predicted class : {class_labels[class_index]}")
        st.write(f"Confidence : {confidence:.2%}")
