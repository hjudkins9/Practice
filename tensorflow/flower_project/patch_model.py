import os
import sys

MODEL_JSON_PATH = 'web_model/model.json'

def main_patch():
    print(f"Attempting to patch file: {MODEL_JSON_PATH}")

    if not os.path.exists(MODEL_JSON_PATH):
        print(f"❌ ERROR: File not found at {MODEL_JSON_PATH}")
        print("   Please run 'python convert.py' first to create it.")
        sys.exit(1)

    try:
        # Read the broken file
        with open(MODEL_JSON_PATH, 'r') as f:
            content = f.read()
        
        # Check if the patch is needed
        if '"batch_shape"' in content:
            print("   Found buggy 'batch_shape' key. Replacing...")
            
            # Replace the first instance (the one in the InputLayer)
            content = content.replace('"batch_shape"', '"batch_input_shape"', 1)
            
            # Write the fixed content back to the file
            with open(MODEL_JSON_PATH, 'w') as f:
                f.write(content)
            
            print("✅ Patch applied successfully.")
        
        elif '"batch_input_shape"' in content:
            print("✅ File is already correct. No patch needed.")
        
        else:
            print("❌ ERROR: Could not find 'batch_shape' or 'batch_input_shape'.")

    except Exception as e:
        print(f"❌ An error occurred during patching: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main_patch()