import my_warnings
import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
import pathlib


# Downloading the flower photos from tensorflow 
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
archive = tf.keras.utils.get_file(origin=dataset_url, extract=True)

# Finding the directory with the flower photos
data_dir = pathlib.Path(archive)

image_count = len(list(data_dir.glob('*/*/*.jpg')))
print(image_count)

print(data_dir)

roses = list(data_dir.glob('*/roses/*'))
PIL.Image.open(str(roses[0]))