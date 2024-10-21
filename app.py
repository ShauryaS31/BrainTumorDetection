import streamlit as st
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout
from tensorflow.keras.applications import VGG19
from PIL import Image
import numpy as np
import pandas as pd
import threading
import pyttsx3


# Function to build the model architecture
def build_model():
    base_model = VGG19(include_top=False, input_shape=(240, 240, 3))
    x = base_model.output
    flat = Flatten()(x)
    class_1 = Dense(4608, activation='relu')(flat)
    drop_out = Dropout(0.2)(class_1)
    class_2 = Dense(1152, activation='relu')(drop_out)
    output = Dense(2, activation='softmax')(class_2)
    model = Model(inputs=base_model.inputs, outputs=output)
    return model

# Load the model and weights
def load_model_with_weights():
    model = build_model()
    try:
        model.load_weights('vgg_unfrozen.weights.h5')
        st.success("Model and weights loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {e}")
    return model

# Streamlit app layout
st.set_page_config(page_title="Brain Tumor Classification", layout="wide")

# Sidebar for instructions
st.sidebar.title("Instructions")
st.sidebar.write("""
Upload an image of the brain to classify whether it has a tumor or not.
Supported formats: JPG, PNG, JPEG.
""")

# Initialize session state for upload history if not already done
if 'upload_history' not in st.session_state:
    st.session_state.upload_history = []

# Load the model
model = load_model_with_weights()

# Define function to preprocess the uploaded image for prediction
def preprocess_image(image):
    img = image.resize((240, 240))  # Resize to the input size of the model
    img_array = np.array(img)

    if img_array.ndim == 2:  # Grayscale image
        img_array = np.stack((img_array,) * 3, axis=-1)  # Convert to RGB
    elif img_array.shape[-1] == 4:  # RGBA image
        img_array = img_array[..., :3]  # Convert to RGB

    img_array = img_array / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Define function to classify the image and return the result
def classify_image(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    class_labels = ['No Tumor', 'Tumor']
    predicted_class = class_labels[np.argmax(prediction)]
    return predicted_class, prediction

# Image upload section
st.title("Brain Tumor Classification")
uploaded_image = st.file_uploader("Upload an image for classification", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    st.write("Image uploaded successfully!")
    image = Image.open(uploaded_image)
    image = image.resize((160, 160))
    st.image(image, caption="Uploaded Image", use_column_width=False)
    
    # Perform classification
    result, prediction = classify_image(image)
    
    # Display the result
    st.subheader("Classification Result")
    if result == "Tumor":
        st.markdown(f"<h3 style='color: red;'>Predicted Class: **{result}**</h3>", unsafe_allow_html=True)
    else:
        st.markdown(f"<h3 style='color: green;'>Predicted Class: **{result}**</h3>", unsafe_allow_html=True)

    st.write(f"Prediction Confidence: {np.max(prediction):.2f}")

    # Save the upload history
    st.session_state.upload_history.append((uploaded_image.name, result, np.max(prediction)))

# Display upload history in the sidebar
st.sidebar.subheader("Upload History")
for img_name, label, confidence in st.session_state.upload_history:
    st.sidebar.write(f"**Image:** {img_name}")
    st.sidebar.write(f"**Result:** {label}")
    st.sidebar.write(f"**Confidence:** {confidence:.2f}")
    st.sidebar.write("---------------")

# Download results option
if st.sidebar.button("Download History"):
    history_df = pd.DataFrame(st.session_state.upload_history, columns=["Image", "Result", "Confidence"])
    history_df.to_csv("upload_history.csv", index=False)
    st.sidebar.success("History saved as upload_history.csv!")

# Additional information section
with st.expander("More Information"):
    st.write("This model classifies brain images into 'Tumor' and 'No Tumor'.")
    st.image("https://via.placeholder.com/150", caption="Model Illustration")  # Replace with actual image path
    st.write("### About the Model")
    st.write("""
    This model is based on VGG19 architecture and has been trained to differentiate between brain tumor images.
    For best results, ensure the uploaded images are clear and well-cropped.
    """)

# Accessibility features (Optional)
if st.sidebar.checkbox("Text-to-Speech for Results"):
    def speak_text(text):
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()

    text_to_speak = f"The predicted class is {result} with a confidence of {np.max(prediction):.2f}."
    threading.Thread(target=speak_text, args=(text_to_speak,)).start()