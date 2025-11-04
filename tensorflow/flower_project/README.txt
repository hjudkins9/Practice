The Flora Finder: A Flower Classification App

This is a web-based AI project that uses a Convolutional Neural Network (CNN) trained in TensorFlow to identify five types of flowers (daisies, dandelions, roses, sunflowers, and tulips) from an image uploaded by a user.

The classification is performed 100% in the user's browser using TensorFlow.js, making it a fast and private-by-default application.

(Add a screenshot of your working application here)

Technology Stack

Backend (Model Training): Python, TensorFlow (Keras)

Frontend (Web App): HTML, Tailwind CSS, React (via in-browser Babel), TensorFlow.js

Conversion Workflow: tensorflowjs_converter (via command-line), custom Python scripts (convert.py, patch_model.py)

Features

Identifies 5 types of flowers: Daisy, Dandelion, Roses, Sunflowers, and Tulips.

Interactive web interface with drag-and-drop image uploading.

Real-time classification performed on the client-side (in the browser).

Responsive design that works on different screen sizes.

The Journey: From Python to the Browser

This project was a complete end-to-end journey, from gathering data and training a model in Python to deploying it in a live web application.

1. The Backend (Model Training)

The core CNN model was built and trained in Python using TensorFlow and Keras. The architecture and training process were based on the official TensorFlow image classification guides.

The model is a standard Sequential CNN containing Conv2D, MaxPooling2D, Dropout, and Dense layers. It was trained on the "flower_photos" dataset to classify the 5 types of flowers.

2. The Frontend (The Web App)

The user interface was built as a single-page index.html file. While the core logic and project idea were my own, the frontend implementation was primarily developed in collaboration with Google's Gemini. This included:

Structuring the app using React (loaded in the browser via Babel).

Styling the components with Tailwind CSS to create a modern, responsive layout.

Writing the JavaScript logic to handle file uploads, manage application state (loading, result, etc.), and trigger the TensorFlow.js model.

3. The Challenge: Project-Defining Conversion Bugs

The most difficult part of this project was not training the model or building the UI; it was getting the trained Python model to run in the browser. The tensorflowjs_converter tool, which is necessary for this process, was buggy and required a custom workflow to fix.

The Overfitting Trade-off

The final model achieved a training accuracy of 99.6% but a validation accuracy of 67.6%.

This is a classic case of overfitting: the model "memorized" the training images very well but was less effective at generalizing to new, unseen images.

This happened because a key feature, Data Augmentation (RandomFlip, RandomRotation), had to be removed. The tensorflowjs_converter could not read these layers, and their inclusion caused a fatal Unknown layer error in the browser.

The project required a trade-off: remove augmentation to make the model functional in the browser, at the known cost of reducing its real-world accuracy.

The "Two-Bug" Patch

Even after removing augmentation, the converter still produced a broken model.json file. Two custom patch scripts were created to solve this:

convert.py: This script runs the standard tensorflowjs_converter to convert the .h5 file, knowing it will fail.

patch_model.py: This script runs after the converter and performs two "find-and-replace" operations on the model.json file to fix its two critical bugs:

Bug 1: It replaces the key "batch_shape" with "batch_input_shape" to prevent the model from failing to load.

Bug 2: It removes the "sequential/" prefix from all layer names to fix the no target variable weight-loading error.

This 3-step pipeline (train -> convert -> patch) was the final solution to get a functional and accurate model.

How to Run This Project

Clone this repository.

Ensure you have Python and the required libraries:

pip install tensorflow tensorflowjs h5py


Train the Model: Run the training script (this will take time for 50 epochs).

python train_model.py


Convert the Model: Run the converter script.

python convert.py


Patch the Model: Run the patch script to fix the converter's bugs.

python patch_model.py


Start the Server:

python -m http.server 8000


Open http://localhost:8000/index.html in your browser.

Future Improvements

While this project is a successful end-to-end proof-of-concept, the 67.6% accuracy is the main area for improvement.

Use Transfer Learning: The #1 priority would be to use a pre-trained model, like MobileNetV2, as the base for the classifier. These models are already optimized for mobile and web, are highly accurate, and their converters are well-supported, which would solve both the accuracy and conversion-bug problems at the same time.

Offline Augmentation: Instead of using augmentation layers, one could create a script to augment the dataset on disk before training, creating 5-10x more training images. This would also help reduce overfitting without breaking the converter.

Full React Build: The frontend could be rebuilt in a proper React/Vite development environment rather than using in-browser Babel, which is not suitable for a production application.
