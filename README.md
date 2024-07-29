AI Model for Detecting Student Learning Styles (VARK)

#Project Overview
The goal of this project is to develop an AI model that can accurately identify students' learning styles based on the VARK model. The model is trained over 20 epochs using a dataset categorized by VARK learning styles to improve its accuracy and reliability.


#Key Features
-Learning Style Detection: Classifies learning styles into Visual, Auditory, Reading/Writing, or Kinesthetic.
-Video Prediction: Processes video files to display predicted learning styles on video frames.
-Model Training and Evaluation: Trained using a dataset and evaluated for accuracy.


#Dataset
The project relies on a dataset categorized by VARK learning styles. Ensure that you have the dataset files in the specified directory or update the file paths in the code as needed.
split the dataset to 4 category its (Visual-Aural-Read/write-Kinesthetic)


#Prerequisites
Make sure you have the following software installed:

Python 3.x
TensorFlow
OpenCV
NumPy
Matplotlib


#Training the Model:

Data Preparation:
The script defines paths to the image dataset.
image_dataset_from_directory function is used to load images and their corresponding labels.
Data is split into training and validation sets (80%/20%).
Images are resized to a fixed size (150x150 pixels) and converted to tensors.
Labels are one-hot encoded for categorical crossentropy loss.
Prefetching is used for performance optimization.

Model Definition:
A Sequential model is built with convolutional and pooling layers for feature extraction.
Flatten layer transforms the extracted features into a 1D vector.
Dense layers with ReLU activation are used for classification.
Dropout layer helps prevent overfitting.
The output layer uses softmax activation with the number of units matching the number of classes (learning styles).

Model Compilation:
Adam optimizer is used for training.
Categorical crossentropy loss is used for multi-class classification.
Accuracy metric is used to evaluate model performance.

Early Stopping:
An EarlyStopping callback is used to stop training if validation accuracy doesn't improve for a certain number of epochs.
This helps prevent overfitting and saves training time.

Training:
The model is trained on the prepared data for a specified number of epochs (default: 20).
Training and validation accuracy/loss are tracked during training.


