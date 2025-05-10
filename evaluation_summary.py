import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# === Paths ===
val_dir = "valid"
model_path = "models/model_efficientnet.keras"
img_size = (128, 128)
batch_size = 32

# === 8 Selected Classes ===
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

# === Load validation set ===
val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    label_mode="categorical",
    class_names=class_names,
    image_size=img_size,
    batch_size=batch_size,
    shuffle=False
)

val_ds = val_ds.map(lambda x, y: (tf.keras.applications.efficientnet.preprocess_input(x), y))

# === Load model ===
model = tf.keras.models.load_model(model_path)

# === Evaluate model ===
loss, acc = model.evaluate(val_ds)
print(f"\n✅ Validation Accuracy: {acc*100:.2f}%")
print(f"✅ Validation Loss: {loss:.4f}")

# === Predictions & Metrics ===
y_true = np.concatenate([y.numpy() for _, y in val_ds])
y_true_labels = np.argmax(y_true, axis=1)

y_pred_probs = model.predict(val_ds)
y_pred_labels = np.argmax(y_pred_probs, axis=1)

# === Classification Report ===
print("\n🧾 Classification Report:")
print(classification_report(y_true_labels, y_pred_labels, target_names=class_names))

cm = confusion_matrix(y_true_labels, y_pred_labels)
plt.figure(figsize=(10, 8))  # Bigger figure
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

# Customize plot
disp.plot(xticks_rotation=45, cmap="Blues", values_format='d')
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=7)
plt.title("Confusion Matrix", fontsize=14, pad=20)
plt.tight_layout()
plt.show()