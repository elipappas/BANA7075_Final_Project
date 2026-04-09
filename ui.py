import streamlit as st
from tensorflow import keras
from utils import prepare_image_for_model

# to run ui: python -m streamlit run ui.py

# Cache the model so it only loads once when the app starts
@st.cache_resource
def load_my_model():
    return keras.models.load_model("xray_cnn_model.keras")

cnn = load_my_model()

st.write("""
# X-Ray Imaging
""")

image = st.file_uploader("Upload X-Ray Image", type=["jpg", "jpeg", "png"])

# Wait for the upload before running the model
if image is not None:
    
    # Show the user the image they just uploaded
    st.image(image, caption="Uploaded X-Ray")
    
    # Clean the image
    cleaned_image = prepare_image_for_model(image)
    
    # Make the prediction (.predict() is the standard Keras method)
    prediction = cnn.predict(cleaned_image)
    st.write(f"Raw Prediction: {prediction}")