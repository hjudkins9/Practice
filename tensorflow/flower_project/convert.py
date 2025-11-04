# Henry Judkins
# This script converts the trained Keras model into the 
# TensorFlow.js format needed for the web app.

import os
import shutil
import subprocess
import sys

# looking for the .h5 file created by our new training script
INPUT_MODEL_PATH = 'saved_models/flower_classifier_model.h5'
# output directory for the converted web model files (model.json, .bin)
WEB_OUTPUT_DIR = 'web_model'

def main_converter():
    # checking if the trained model file exists
    if not os.path.exists(INPUT_MODEL_PATH):
        print(f"Error: {INPUT_MODEL_PATH} not found.")
        print("   Please run 'python train_model.py' first.")
        sys.exit(1)

    # cleaning up the old web_model directory
    if os.path.exists(WEB_OUTPUT_DIR):
        print(f"Removing old '{WEB_OUTPUT_DIR}' folder...")
        shutil.rmtree(WEB_OUTPUT_DIR)
    # creating a new directory for the new model files
    os.makedirs(WEB_OUTPUT_DIR)

    # running the conversion
    print(f"\nConverting '{INPUT_MODEL_PATH}' to TensorFlow.js format...")
    try:
        # calling the command-line tool directly
        subprocess.run([
            'tensorflowjs_converter',
            '--input_format=keras',    # telling it we're using a Keras file
            INPUT_MODEL_PATH,          # The input .h5 file
            WEB_OUTPUT_DIR             # The output folder
        ], check=True) # 'check=True' makes the script fail if the command fails
        
        print("\nConversion Complete.")

    except Exception as e:
        print(f"\nA critical error occurred during conversion:")
        print(e)
        sys.exit(1)

if __name__ == '__main__':
    main_converter()