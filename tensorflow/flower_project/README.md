# The Flora Finder: A Flower Classification App

This is a web-based AI project that uses a Convolutional Neural Network (CNN) trained in TensorFlow to identify five types of flowers — *daisies, dandelions, roses, sunflowers,* and *tulips* — from an image uploaded by a user.

The classification is performed 100% in the user's browser using TensorFlow.js, making it a fast and privacy-friendly application.

<img src="flora_finder.png" alt="Flora Finder Project">

---

## Technology Stack

**Backend (Model Training):**  
- Python  
- TensorFlow (Keras)

**Frontend (Web App):**  
- HTML  
- Tailwind CSS  
- React (via in-browser Babel)  
- TensorFlow.js  

**Conversion Workflow:**  
- `tensorflowjs_converter` (via command-line)  
- Custom Python scripts (`convert.py`, `patch_model.py`)

---

## Features

- Identifies 5 types of flowers: *Daisy, Dandelion, Rose, Sunflower,* and *Tulip*  
- Interactive web interface with drag-and-drop image uploading  
- Real-time classification performed on the client-side (in the browser)  
- Responsive design that works across several different displays  

---

### The Backend (Model Training)

The core CNN model was built and trained in Python using TensorFlow and Keras.  
The architecture and training process followed the official TensorFlow image classification guides.

The model is a Sequential CNN containing `Conv2D`, `MaxPooling2D`, `Dropout`, and `Dense` layers.  
It was trained on the *flower_photos* dataset to classify the five types of flowers.

---

### The Frontend (The Web App)

The user interface was built as a single-page `index.html` file.  
While the core logic and project concept were original, the frontend was developed in collaboration with Google’s Gemini, which helped with:

- Structuring the app using React (loaded via in-browser Babel)  
- Styling components with Tailwind CSS for a modern, responsive look  
- Writing JavaScript logic for file uploads, application state management, and TensorFlow.js inference  

---

### The Challenge: Project-Defining Conversion Bugs

The hardest part of the project wasn’t model training or UI building — it was getting the trained Python model to run in the browser.  

The `tensorflowjs_converter` tool, required for this step, was buggy and needed a custom workflow to make it work properly.

---

## The Overfitting Trade-Off

The final model achieved:

- **Training Accuracy:** 99.6%  
- **Validation Accuracy:** 67.6%

This reflects a common symptom of overfitting — the model memorized training data well but struggled to generalize to unseen images.

This was due to removing data augmentation layers (`RandomFlip`, `RandomRotation`), as the converter could not process them.  
Their inclusion caused a fatal `"Unknown layer"` error in TensorFlow.js.

**Trade-off:**  
Removing augmentation allowed the model to run in-browser, but reduced real-world accuracy.

---

## The "Two-Bug" Patch

Even without augmentation, the converter still produced a broken `model.json`.  
Two custom patch scripts fixed this:

1. **`convert.py`**  
   - Runs the standard `tensorflowjs_converter`, knowing it will fail.

2. **`patch_model.py`**  
   - Fixes the generated `model.json` with two replacements:
     - **Bug 1:** Replace `"batch_shape"` → `"batch_input_shape"` to fix loading errors.  
     - **Bug 2:** Remove `"sequential/"` prefixes from all layer names to fix weight-loading issues.

**Final workflow:**
`train → convert → patch`

---

## How to Run This Project

1. **Clone the repository**

2. **Install dependencies**
   ```bash
   pip install tensorflow tensorflowjs h5py
3. **Train the Model**
   ```bash
   python train_model.py
4. **Convert the Model**
   ```bash
   python convert.py
5. **Patch the Model**
   ```bash
   python patch_model.py
6. **Start a Local Server**
   ```bash
   python -m http.server 8000
7. **Open in Browser**
   Go to http://localhost:8000/index.html
   
---

## Future Improvements
While this project is a successfull end-to-end proof of concept, improving accuracy is the next big step. 

1. **Use Transfer Learning**
   Adopt a pre-trained model such as MobileNetV2 as the base classifier.
2. **Offline Augmentation**
   Instead of using augmentation layers, pre-augmen tthe dataset on disk to create 5-10x more training samples.
   This mitigates overfitting without breaking the converter.
3. **Full React Build**
   Rebuild the frontend using a proper React + Vite development setup instead of in-browser Babel for a production-ready environment.

---
