import os
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

# --- Model Loading (Cached for performance) ---
#@st.cache_resource
@st.cache(allow_output_mutation=True)
def load_keras_model():
    """Loads the pre-trained Keras model and caches it."""
    try:
        model = tf.keras.models.load_model('modelf.h5')
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

model = load_keras_model()
class_names = ['Not Forged', 'Forged']

# --- Streamlit UI ---
st.title("🔎 Image Forgery Detector")
st.write(
    "Upload an image to determine if it has been digitally manipulated. "
    "This app uses **Error Level Analysis (ELA)** to identify potential forgeries."
)

uploaded_file = st.file_uploader("Choose an image file", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None and model is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    st.write("")

    with st.spinner('Analyzing the image...'):
        try:
            # Open the image using Pillow
            original_image = Image.open(uploaded_file).convert('RGB')

            # 1. Perform ELA
            ela_image = ELA(original_image)
            
            # 2. Prepare the image for the model
            ela_image_resized = ela_image.resize((420, 420))
            image_array = np.array(ela_image_resized)
            image_array = image_array.reshape(-1, 420, 420, 3)

            # 3. Make a prediction
            prediction = model.predict(image_array)
            predicted_class_index = np.argmax(prediction, axis=1)[0]
            predicted_class_name = class_names[predicted_class_index]
            confidence_score = np.max(prediction) * 100

            # 4. Display the result
            st.subheader("Analysis Result")
            if predicted_class_name == 'Forged':
                st.error(f"Prediction: **{predicted_class_name}**")
            else:
                st.success(f"Prediction: **{predicted_class_name}**")

            st.metric(label="Confidence Score", value=f"{confidence_score:.2f}%")
            
            # Display the ELA image for user inspection
            st.image(ela_image, caption='Error Level Analysis (ELA) Image', use_column_width=True)
            st.info(
                "**How to interpret the ELA image:** In authentic images, the ELA result is "
                "typically uniform and dark. In forged images, edited areas often appear "
                "significantly brighter than the rest of the image."
            )

        except Exception as e:
            st.error(f"An error occurred during analysis: {e}")

elif model is None:
    st.warning("The prediction model could not be loaded. Please ensure 'modelf.h5' is in the correct directory.")