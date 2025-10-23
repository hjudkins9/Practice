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
        
        original_content = content
        was_patched = False

        # --- FIX 1: The batchInputShape bug ---
        if '"batch_shape"' in content:
            print("   PATCH 1: Found 'batch_shape'. Replacing with 'batch_input_shape'...")
            content = content.replace('"batch_shape"', '"batch_input_shape"', 1)
            was_patched = True
        
        # --- FIX 2: The weight name prefix bug ---
        if '"sequential/' in content:
            print("   PATCH 2: Found 'sequential/' prefix. Removing all instances...")
            # Replace all instances of this prefix
            content = content.replace('"sequential/', '"') 
            was_patched = True

        # Save the file only if we made changes
        if was_patched:
            with open(MODEL_JSON_PATH, 'w') as f:
                f.write(content)
            print("✅ Patch(es) applied successfully.")
        else:
            print("✅ File appears to be correct already. No patches needed.")

    except Exception as e:
        print(f"❌ An error occurred during patching: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main_patch()
    print("\n------------------------------------------------------------")
    print("Patch complete. You can now start your HTTP server.")
    print("------------------------------------------------------------")