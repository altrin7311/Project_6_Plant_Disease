import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess

def predict_image(img_path, model_type, class_names):
    # Select model and preprocessing
    if model_type == "efficientnet":
        model_path = "models/best_model.keras"
        preprocess_func = efficientnet_preprocess
    elif model_type == "densenet":
        model_path = "models/model_densenet.keras"
        preprocess_func = densenet_preprocess
    else:
        raise ValueError("Unsupported model type. Choose 'efficientnet' or 'densenet'.")

    # Load model
    model = load_model(model_path)

    # Load and preprocess image
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_func(img_array)

    # Predict
    preds = model.predict(img_array)[0]
    top_idx = np.argmax(preds)
    confidence = preds[top_idx]

    return class_names[top_idx], float(confidence)