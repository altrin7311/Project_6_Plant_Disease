import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing import image
from PIL import Image
import io

# === Load model ===
model = tf.keras.models.load_model("models/model_efficientnet.keras")

# === Class names (8 selected)
class_names = [
    "Apple___Cedar_apple_rust",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Grape___Black_rot",
    "Potato___Late_blight",
    "Tomato___Early_blight",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___healthy"
]

# === Streamlit UI ===
st.set_page_config(page_title="Crop Disease Detector", layout="centered")
st.title("🌿 Plant Disease Prediction")
st.write("Upload a crop leaf image (JPG, PNG, WEBP) and get instant prediction.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file:
    try:
        # === Display image ===
        image_bytes = uploaded_file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        st.image(img, caption="Uploaded Image", use_column_width=True)

        # === Preprocess ===
        img_resized = img.resize((128, 128))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # === Predict ===
        preds = model.predict(img_array)[0]
        pred_class = class_names[np.argmax(preds)]
        confidence = float(np.max(preds))

        # === Display result ===
        st.success(f"🧠 Predicted: **{pred_class}** with **{confidence*100:.2f}%** confidence.")

    except Exception as e:
        st.error(f"⚠️ Error processing image: {str(e)}")
