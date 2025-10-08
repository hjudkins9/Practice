import tensorflow as tf
from tensorflow import keras
import os
import sys
# CRITICAL ADDITION: Import the TensorFlow.js converter utility
import tensorflowjs as tfjs

# --- Configuration ---
INPUT_MODEL_PATH = 'saved_models/flower_classifier_model.keras'
OUTPUT_MODEL_PATH = 'saved_models/flower_classifier_model_fixed.keras'
# Define the output path for the web model files
WEB_OUTPUT_DIR = 'web_model'
# ---------------------

# --- CRITICAL: MODEL STRUCTURE DEFINITION ---
# IMPORTANT: This block is used ONLY if direct tf.keras.models.load_model() fails.
def create_manual_model_structure():
    """
    PASTE YOUR MODEL DEFINITION HERE: This code re-creates the architecture
    so the saved weights can be loaded correctly if the original file lacks structure.
    
    CRITICAL FIX: Temporarily remove data augmentation layers for conversion stability.
    """
    
    # ------------------ START REQUIRED EDIT (Cleaned for conversion stability) ------------------
    # NOTE: The explicit tf.keras.layers.Input is REMOVED to avoid the batch_shape error during conversion.
    # The input shape is instead passed to the first actual processing layer (Rescaling).
    model = keras.Sequential([
        # 1. Preprocessing and Convolutional Base (STARTING from Rescaling)
        # We define the input shape here to ensure the model has structure.
        tf.keras.layers.Rescaling(1./255, input_shape=(180, 180, 3)), 
        
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.2), 
        tf.keras.layers.Conv2D(128, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        
        # 2. Classifier Head
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        # Output layer must have 'softmax' activation for probability output
        tf.keras.layers.Dense(5, activation='softmax') 
    ])
    
    return model
    # ------------------ END REQUIRED EDIT --------------------
# --- END MODEL STRUCTURE DEFINITION ---

def resave_model():
    """Loads the original model, resaves it in Keras native format, and then converts it directly to TF.js."""
    
    if not os.path.exists(INPUT_MODEL_PATH):
        print(f"Error: Input model not found at path: {INPUT_MODEL_PATH}")
        sys.exit(1)

    # --- Attempt 1: Direct Load ---
    try:
        print(f"Attempt 1: Loading full model structure from {INPUT_MODEL_PATH}...")
        model = tf.keras.models.load_model(INPUT_MODEL_PATH)
        print("Model loaded successfully via direct load.")
    except Exception as e_direct:
        # --- Attempt 2: Manual Structure and Load Weights ---
        print(f"Direct load failed. Attempting to load using manual structure definition.")
        print(f"Error during direct load was: {e_direct}")
        
        try:
            model = create_manual_model_structure()
            model.load_weights(INPUT_MODEL_PATH)
            print("Weights loaded successfully into manual structure.")
            
        except Exception as e_manual:
            print("\n❌ CRITICAL FAILURE: Could not load the model even with manual structure.")
            print("   -> Your manual model definition (layers) does not match the weights in the .keras file.")
            print(f"   -> Detailed Manual Load Error: {e_manual}")
            sys.exit(1)


    # --- Save the Compatible Native Keras File (Optional, but good practice) ---
    try:
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        print(f"Saving full model structure and weights to: {OUTPUT_MODEL_PATH}")
        model.save(OUTPUT_MODEL_PATH)
        print(f"Keras native model saved to: {OUTPUT_MODEL_PATH}")
        
    except Exception as e_save:
        print(f"\n❌ A critical error occurred during saving the Keras native model:")
        print(e_save)
        sys.exit(1)
        
    # --- CRITICAL: DIRECT TF.JS CONVERSION ---
    print(f"\nConverting model directly to TensorFlow.js Layers Model format in directory: {WEB_OUTPUT_DIR}")
    try:
        os.makedirs(WEB_OUTPUT_DIR, exist_ok=True) # Ensure the output directory exists
        
        # This function is more stable than the command line tool
        tfjs.converters.save_keras_model(model, WEB_OUTPUT_DIR)
        
        print("\n✅ TensorFlow.js Conversion Complete!")
        print(f"Files are now in the '{WEB_OUTPUT_DIR}' folder, ready for the web app.")
        print("\n------------------------------------------------------------")
        print("NEXT STEP: Please run 'python check_model_files.py' to confirm conversion,")
        print("then restart your HTTP server if it is not already running.")
        print("------------------------------------------------------------")
        
    except Exception as e_tfjs:
        print(f"\n❌ A critical error occurred during the direct TensorFlow.js conversion:")
        print(e_tfjs)
        sys.exit(1)


if __name__ == '__main__':
    resave_model()
