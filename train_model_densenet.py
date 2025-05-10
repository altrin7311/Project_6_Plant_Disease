import os
import tensorflow as tf
from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input
from tensorflow.keras import layers, models, callbacks

# === Paths ===
train_dir = "train"
val_dir = "valid"
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "model_densenet.keras")

# === Parameters ===
img_size = (128, 128)
batch_size = 32
epochs = 15

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

# === Load datasets ===
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    label_mode="categorical",
    class_names=class_names,
    image_size=img_size,
    batch_size=batch_size,
    shuffle=True
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    label_mode="categorical",
    class_names=class_names,
    image_size=img_size,
    batch_size=batch_size,
    shuffle=False
)

# === Apply DenseNet preprocessing ===
train_ds = train_ds.map(lambda x, y: (preprocess_input(x), y))
val_ds = val_ds.map(lambda x, y: (preprocess_input(x), y))

# === Build model ===
base_model = DenseNet121(weights="imagenet", include_top=False, input_shape=img_size + (3,))
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(len(class_names), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# === Callbacks ===
checkpoint_cb = callbacks.ModelCheckpoint(model_path, save_best_only=True, monitor="val_accuracy", mode="max")
earlystop_cb = callbacks.EarlyStopping(patience=3, restore_best_weights=True)

# === Train model ===
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=[checkpoint_cb, earlystop_cb]
)

print(f"\n✅ Model training complete. Best model saved to {model_path}")