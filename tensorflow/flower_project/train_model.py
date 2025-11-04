# Henry Judkins
# This script trains the flower classification model.
# It downloads the "flower_photos" dataset, builds a clean
# Convolutional Neural Network (CNN) model, and trains it.
# The final, trained model is saved as 'flower_classifier_model.h5'
# in the 'saved_models' directory.

import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
from tensorflow import keras
import pathlib

# CONFIGURATION
batch_size = 32
img_height = 180
img_width = 180
epochs = 50

# DATA SETUP
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
archive = tf.keras.utils.get_file(origin=dataset_url, extract=True)
data_dir = pathlib.Path(archive) / 'flower_photos'

# DATA LOADING
print("Loading data for training and validation...")
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)
class_names = train_ds.class_names
print(f"Class names recognized by the model: {class_names}")

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
num_classes = len(class_names)

# MODEL DEFINITION
print("Building clean Sequential model...")

model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.2), 
    tf.keras.layers.Conv2D(128, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(128, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(num_classes) # Output logits
])

model.summary()

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

# TRAINING
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# MODEL SAVING
KERAS_OUTPUT_FILE = 'flower_classifier_model.h5' 
export_dir = os.path.join('saved_models', KERAS_OUTPUT_FILE)
os.makedirs('saved_models', exist_ok=True)

model.save(export_dir) 
print(f"\nKeras Model saved to: {export_dir}")

# SAVING CLASS NAMES
class_names_path = os.path.join('saved_models', 'class_names.txt')
with open(class_names_path, 'w') as f:
    f.write('\n'.join(class_names))
print(f"Class names saved to: {class_names_path}")

print("\nTraining complete. Run 'python convert.py' to create the web model.")