# Henry Judkins
# Using the tensorflow dataset/model to identify different kinds of flowers

import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
import pathlib
import matplotlib.pyplot as plt

# Downloading the flower photos from tensorflow 
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
archive = tf.keras.utils.get_file(origin=dataset_url, extract=True)

# Finding the directory with the flower photos
data_dir = pathlib.Path(archive)

# Updating data_dir to point to the correct sub-directory
data_dir = data_dir / 'flower_photos'

# Defining parameters for the loader
batch_size = 32
img_height = 180
img_width = 180

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
print(f"Class names: {class_names}")

AUTOTUNE = tf.data.AUTOTUNE

# Apply caching and prefetching for performance
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

num_classes = 5

# Create a data augmentation layer
data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),
  tf.keras.layers.RandomRotation(0.1),
  tf.keras.layers.RandomZoom(0.1),
])

# Define the new, deeper model with dropout
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

# It is recommended to train for more than 3 epochs to see improvement
 epochs = 50
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# Define the directory and filename where you will save the model
# FIX 1: Added the .keras extension for proper Keras model saving
export_dir = 'flower_project/saved_models/flower_classifier_model.keras'

# FIX 2: Ensure the parent directory ('saved_models') exists before saving the file
os.makedirs(os.path.dirname(export_dir), exist_ok=True)

# Save the entire model (architecture, weights, optimizer state)
model.save(export_dir) 
print(f"\nModel saved to: {export_dir}")

# Also, save the class_names list, as you'll need it for prediction in the web app
class_names_path = 'saved_models/class_names.txt'
os.makedirs(os.path.dirname(class_names_path), exist_ok=True)
with open(class_names_path, 'w') as f:
    f.write('\n'.join(class_names))
print(f"Class names saved to: {class_names_path}")

"""
# Load and preprocess a single image
image_path = 'flower_pictures/download.jpg'
img = tf.keras.utils.load_img(
    image_path,
    target_size=(img_height, img_width)
)

# Convert the image to an array and add a batch dimension
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Shape becomes (1, 180, 180, 3)

# Use the predict method to get the output from the model
predictions = model.predict(img_array)

# The output 'predictions' is a NumPy array of logits
# Apply softmax to get probabilities
score = tf.nn.softmax(predictions[0])

# Get the predicted class and its confidence
predicted_class_index = np.argmax(score)
predicted_class_name = class_names[predicted_class_index]
confidence = 100 * np.max(score)

print(f"This image most likely belongs to {predicted_class_name} with a {confidence:.2f}% confidence.")
"""