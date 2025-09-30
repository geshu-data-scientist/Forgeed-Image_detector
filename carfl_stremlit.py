import os
import requests
import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image, ImageChops

# --- Page Configuration ---
st.set_page_config(
    page_title="Image Forgery Detector",
    page_icon="🔎",
    layout="centered"
)

# --- Model Handling ---
# IMPORTANT: Replace this with the actual direct download link to your model
MODEL_URL = "YOUR_DIRECT_DOWNLOAD_URL_HERE"
MODEL_PATH = "modelf.h5"

def download_file(url, file_path):
    """Downloads a file from a URL to a local path with a progress bar."""
    if os.path.exists(file_path):
        return True # File already exists

    st.info("Model not found locally. Downloading from the web...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Check for download errors

        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 KB

        progress_bar = st.progress(0, text="Downloading model...")
        with open(file_path, "wb") as f:
            downloaded_size = 0
            for data in response.iter_content(block_size):
                downloaded_size += len(data)
                f.write(data)
                # Update progress bar
                progress = min(downloaded_size / total_size, 1.0)
                progress_bar.progress(progress, text=f"Downloading model... {int(progress*100)}%")
        
        progress_bar.empty() # Remove the progress bar after completion
        st.success("Model downloaded successfully!")
        return True
    except requests.exceptions.RequestException as e:
        st.error(f"Error downloading model: {e}")
        st.error("Please ensure the download URL is correct and you have an internet connection.")
        return False

# --- Helper Function (ELA) ---
def ELA(img, quality=90):
    """Performs Error Level Analysis on a Pillow image object."""
    TEMP = 'ela_temp.jpg'
    SCALE = 10
    
    # Re-save the image at a specific quality
    img.save(TEMP, 'JPEG', quality=quality)
    temporary = Image.open(TEMP)
    
    # Find the difference between the original and the re-saved image
    diff = ImageChops.difference(img, temporary)
    
    # Scale the difference to make it more visible
    d = diff.load()
    WIDTH, HEIGHT = diff.size
    for x in range(WIDTH):
        for y in range(HEIGHT):
            d[x, y] = tuple(k * SCALE for k in d[x, y])
            
    # Clean up the temporary file
    os.remove(TEMP)
    return diff

# --- Main App ---
st.title("🔎 Image Forgery Detector")
st.write(
    "Upload an image to determine if it has been digitally manipulated. "
    "This app uses **Error Level Analysis (ELA)** to identify potential forgeries."
)

# Download the model first, then define the loading function and the rest of the app
model_downloaded = download_file(MODEL_URL, MODEL_PATH)
model = None

if model_downloaded:
    #@st.cache_resource
    @st.cache(allow_output_mutation=True)
    def load_keras_model():
        """Loads the pre-trained Keras model from the local path."""
        try:
            loaded_model = tf.keras.models.load_model(MODEL_PATH)
            return loaded_model
        except Exception as e:
            st.error(f"Error loading the model from disk: {e}")
            return None

    model = load_keras_model()

# --- Streamlit UI ---
class_names = ['Not Forged', 'Forged']
uploaded_file = st.file_uploader("Choose an image file", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    if model is not None:
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
        st.write("")

        with st.spinner('Analyzing the image...'):
            try:
                original_image = Image.open(uploaded_file).convert('RGB')
                ela_image = ELA(original_image)
                
                # Prepare image for the model
                ela_image_resized = ela_image.resize((420, 420))
                image_array = np.array(ela_image_resized)
                image_array = image_array.reshape(-1, 420, 420, 3)

                # Make prediction
                prediction = model.predict(image_array)
                predicted_class_index = np.argmax(prediction, axis=1)[0]
                predicted_class_name = class_names[predicted_class_index]
                confidence_score = np.max(prediction) * 100

                # Display result
                st.subheader("Analysis Result")
                if predicted_class_name == 'Forged':
                    st.error(f"Prediction: **{predicted_class_name}**")
                else:
                    st.success(f"Prediction: **{predicted_class_name}**")

                st.metric(label="Confidence Score", value=f"{confidence_score:.2f}%")
                
                st.image(ela_image, caption='Error Level Analysis (ELA) Image', use_column_width=True)
                st.info(
                    "**How to interpret the ELA image:** In authentic images, the ELA result is "
                    "typically uniform and dark. In forged images, edited areas often appear "
                    "significantly brighter than the rest of the image."
                )

            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")
    else:
        st.warning("The prediction model could not be loaded. Please check the logs for errors.")
