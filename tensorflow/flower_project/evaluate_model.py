import tensorflow as tf
import pathlib
import os

# --- CONFIGURATION (Must match train_model.py) ---
batch_size = 32
img_height = 180
img_width = 180
MODEL_PATH = 'saved_models/flower_classifier_model.h5'

# --- 1. POINT TO THE EXISTING DATASET ---
print("Locating the validation dataset...")

# This path is the standard Keras cache directory in your Codespace
# We are assuming train_model.py already downloaded and extracted this.
data_dir = pathlib.Path.home() / '.keras' / 'datasets' / 'flower_photos'

if not data_dir.exists():
    print(f"❌ ERROR: Dataset not found at {data_dir}")
    print("Please run 'python train_model.py' once to download the data.")
    exit()
else:
    print(f"Dataset found at {data_dir}")

# --- 2. PREPARE THE DATASET ---
print("Loading images from directory...")
try:
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        # Must use the same seed as your training script
        seed=123, 
        image_size=(img_height, img_width),
        batch_size=batch_size
    )
except Exception as e:
    print(f"Error loading dataset from {data_dir}: {e}")
    exit()

# Optimize the dataset for performance
AUTOTUNE = tf.data.AUTOTUNE
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# --- 3. LOAD YOUR TRAINED MODEL ---
print(f"Loading trained model from: {MODEL_PATH}")
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file not found at {MODEL_PATH}")
    exit()
    
model = tf.keras.models.load_model(MODEL_PATH)

# --- 4. EVALUATE THE MODEL ---
print("\nEvaluating model accuracy on the validation dataset...")
results = model.evaluate(val_ds)

print("-" * 30)
print(f"Test Loss: {results[0]:.4f}")
print(f"Test Accuracy: {results[1] * 100:.2f}%")
print("-" * 30)