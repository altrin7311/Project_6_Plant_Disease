### with MobileNetV2, 38 classes, 128x128 images, 25 epochs

import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
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
epochs = 25

# ✅ Load and augment training data
train_ds = image_dataset_from_directory(
    train_dir,
    label_mode="categorical",
    image_size=img_size,
    batch_size=batch_size,
    shuffle=True
)

val_ds = image_dataset_from_directory(
    val_dir,
    label_mode="categorical",
    image_size=img_size,
    batch_size=batch_size,
    shuffle=False
)

# ✅ Data preprocessing (normalization for MobileNetV2)
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
val_ds = val_ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

# ✅ Build MobileNetV2-based model
base_model = MobileNetV2(input_shape=img_size + (3,), include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze base model for faster training

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(train_ds.element_spec[1].shape[1], activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ✅ Callbacks
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

# ✅ Train model
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=[checkpoint, early_stop]
)