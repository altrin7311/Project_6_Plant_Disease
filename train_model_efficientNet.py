import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras import layers, models, callbacks
import os

# Paths
train_dir = "train"
val_dir = "valid"
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)

# Parameters
img_size = (128, 128)
batch_size = 32
epochs = 15

# ✅ Selected 8 classes to train on
selected_classes = [
    "Apple___Cedar_apple_rust",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Grape___Black_rot",
    "Potato___Late_blight",
    "Tomato___Early_blight",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___healthy"
]

# ✅ Load training data for selected classes only
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    label_mode="categorical",
    image_size=img_size,
    batch_size=batch_size,
    shuffle=True,
    seed=42,
    class_names=selected_classes
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    label_mode="categorical",
    image_size=img_size,
    batch_size=batch_size,
    shuffle=False,
    class_names=selected_classes
)

# ✅ Preprocessing and performance optimization
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
val_ds = val_ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

# ✅ Build the model using EfficientNetB0
base_model = EfficientNetB0(input_shape=img_size + (3,), include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze base model

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(len(selected_classes), activation='softmax')  # 8 classes
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ✅ Callbacks for saving and early stopping
checkpoint = callbacks.ModelCheckpoint(
    filepath=os.path.join(model_dir, 'best_model.keras'),
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)
early_stop = callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=3,
    restore_best_weights=True
)

# ✅ Train the model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=[checkpoint, early_stop]
)