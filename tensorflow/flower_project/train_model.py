# Henry Judkins
# Script to train a Flower Classifier CNN and convert it for TensorFlow.js web use.

import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import pathlib
# CRITICAL ADDITION: Import the converter tool
# Install this library: pip install tensorflowjs
import tensorflowjs as tfjs 

# --- CONFIGURATION ---
batch_size = 32
img_height = 180
img_width = 180
# The directory where the browser-ready model files will be saved
OUTPUT_DIR = 'web_model' 

# --- 1. DATA SETUP ---

# Downloading the flower photos from tensorflow 
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
archive = tf.keras.utils.get_file(origin=dataset_url, extract=True)

# Finding the directory with the flower photos
data_dir = pathlib.Path(archive)

# Updating data_dir to point to the correct sub-directory
data_dir = data_dir / 'flower_photos'

# --- 2. DATA LOADING ---

print("Loading data for training and validation...")
# Creating the training dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Creating the validation dataset
val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

# Get the class names from the dataset before caching
class_names = train_ds.class_names
print(f"Class names recognized by the model: {class_names}")

AUTOTUNE = tf.data.AUTOTUNE

# Apply caching and prefetching for performance
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

num_classes = len(class_names) # Use dynamic length instead of hardcoded 5

# --- 3. MODEL DEFINITION AND TRAINING ---

# Create a data augmentation layer
data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),
  tf.keras.layers.RandomRotation(0.1),
  tf.keras.layers.RandomZoom(0.1),
])

# Define the Convolutional Neural Network (CNN) model
model = tf.keras.Sequential([
    data_augmentation,
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.2), # Adding a dropout layer
    tf.keras.layers.Conv2D(128, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(128, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(num_classes)
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

# Train for 50 epochs
epochs = 50
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# --- 4. MODEL SAVING AND CONVERSION ---

# A. Save the model in the native Keras format (optional, but good practice)
KERAS_OUTPUT_FILE = 'flower_classifier_model.keras'
# Define the directory and filename where you will save the model
export_dir = os.path.join('saved_models', KERAS_OUTPUT_FILE)

# Ensure the parent directory ('saved_models') exists before saving the file
os.makedirs('saved_models', exist_ok=True)

# Save the entire model (architecture, weights, optimizer state)
model.save(export_dir) 
print(f"\nKeras Model saved to: {export_dir}")

# Also, save the class_names list, as you'll need it for prediction in the web app
class_names_path = os.path.join('saved_models', 'class_names.txt')
os.makedirs(os.path.dirname(class_names_path), exist_ok=True)
with open(class_names_path, 'w') as f:
    f.write('\n'.join(class_names))
print(f"Class names saved to: {class_names_path}")


# B. Convert the model to TensorFlow.js Layers Model format
# This creates the 'web_model/model.json' and 'web_model/weights.bin' files
print(f"\nConverting model to TensorFlow.js format in directory: {OUTPUT_DIR}")
os.makedirs(OUTPUT_DIR, exist_ok=True) # Ensure the output directory exists
# CRITICAL ADDITION: This is the line that creates the files for your web app
tfjs.converters.save_keras_model(model, OUTPUT_DIR)
print("Conversion Complete. Files are now in the 'web_model' folder, ready for App.jsx.")
