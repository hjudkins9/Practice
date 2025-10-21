import os
import shutil
import subprocess
import sys

# We are looking for the .h5 file created by our new training script
INPUT_MODEL_PATH = 'saved_models/flower_classifier_model.h5'
WEB_OUTPUT_DIR = 'web_model'

def main_converter():
    if not os.path.exists(INPUT_MODEL_PATH):
        print(f"❌ Error: {INPUT_MODEL_PATH} not found.")
        print("   Please run 'python train_model.py' first.")
        sys.exit(1)

    if os.path.exists(WEB_OUTPUT_DIR):
        print(f"Removing old '{WEB_OUTPUT_DIR}' folder...")
        shutil.rmtree(WEB_OUTPUT_DIR)
    os.makedirs(WEB_OUTPUT_DIR)

    print(f"\nConverting '{INPUT_MODEL_PATH}' to TensorFlow.js format...")
    try:
        subprocess.run([
            'tensorflowjs_converter',
            '--input_format=keras',
            INPUT_MODEL_PATH,
            WEB_OUTPUT_DIR
        ], check=True)
        
        print("\n✅ Conversion Complete!")

    except Exception as e:
        print(f"\n❌ A critical error occurred during conversion:")
        print(e)
        sys.exit(1)

if __name__ == '__main__':
    main_converter()
    print("\n------------------------------------------------------------")
    print("NEXT STEP: Restart your HTTP server and refresh index.html.")
    print("------------------------------------------------------------")