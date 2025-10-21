# Henry Judkins
# Script to train a Flower Classifier CNN
# This version uses the Keras Functional API for a clean, convertible model.

import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
from tensorflow import keras
import pathlib

# --- CONFIGURATION ---
batch_size = 32
img_height = 180
img_width = 180
# Let's do a quick 2-epoch test first to prove it works
epochs = 2 
#epochs = 50 # You can change this back after the test

# --- 1. DATA SETUP ---
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
archive = tf.keras.utils.get_file(origin=dataset_url, extract=True)
data_dir = pathlib.Path(archive) / 'flower_photos'

# --- 2. DATA LOADING ---
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

# --- 3. MODEL DEFINITION (FUNCTIONAL API) ---
print("Building clean Functional model...")

# 1. Define the input
inputs = keras.Input(shape=(img_height, img_width, 3), name="input_layer")


# 3. Define the rest of the model
x = keras.layers.Rescaling(1./255)(inputs) # Start from 'x', not 'inputs'
x = keras.layers.Conv2D(64, 3, activation='relu')(x)
x = keras.layers.MaxPooling2D()(x)
x = keras.layers.Conv2D(64, 3, activation='relu')(x)
x = keras.layers.MaxPooling2D()(x)
x = keras.layers.Dropout(0.2)(x)
x = keras.layers.Conv2D(128, 3, activation='relu')(x)
x = keras.layers.MaxPooling2D()(x)
x = keras.layers.Conv2D(128, 3, activation='relu')(x)
x = keras.layers.MaxPooling2D()(x)
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(256, activation='relu')(x)

# 4. Define the output layer (for logits)
outputs = keras.layers.Dense(num_classes, name="output_layer")(x)

# 5. Create the model
model = keras.Model(inputs=inputs, outputs=outputs)

model.summary() # This will show you the new structure

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

# --- 4. TRAINING ---
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# --- 5. MODEL SAVING ---
# We will save as .h5, because we know the converter likes it.
KERAS_OUTPUT_FILE = 'flower_classifier_model.h5' 
export_dir = os.path.join('saved_models', KERAS_OUTPUT_FILE)
os.makedirs('saved_models', exist_ok=True)

model.save(export_dir) 
print(f"\nKeras Model saved to: {export_dir}")

# Save class names
class_names_path = os.path.join('saved_models', 'class_names.txt')
with open(class_names_path, 'w') as f:
    f.write('\n'.join(class_names))
print(f"Class names saved to: {class_names_path}")

print("\nTraining complete. Run 'python convert.py' to create the web model.")