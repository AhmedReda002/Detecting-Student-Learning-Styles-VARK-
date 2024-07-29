## AI Model for Detecting Student Learning Styles (VARK)

This repository contains a Python-based AI model designed to classify student learning styles according to the VARK model: Visual, Auditory, Read/Write, and Kinesthetic. The model leverages image analysis techniques and deep learning to analyze visual data and predict the corresponding learning style.

## Key Features

* **Learning Style Classification:** Classifies images into one of the four VARK learning styles.
* **Data Handling:** Utilizes TensorFlow's `image_dataset_from_directory` for loading and preprocessing image data.
* **Model Architecture:** Implements a Convolutional Neural Network (CNN) with layers for feature extraction and classification.
* **Training and Evaluation:** Trains the model using early stopping to avoid overfitting and evaluates performance on a validation set.
* **Visualization:** Plots training and validation accuracy and loss for performance assessment.

## Libraries Used

* **tensorflow:** For building and training the deep learning model.
* **matplotlib:** For plotting training history and visualizing performance.

## Dataset

Ensure your dataset is organized with directories representing each VARK category. For example:

dataset/
Visual/
image1.jpg
image2.jpg
...
Auditory/
image1.jpg
image2.jpg
...
Read/Write/
image1.jpg
image2.jpg
...
Kinesthetic/
image1.jpg
image2.jpg
...
Update the `data_dir` path in the code to point to your dataset directory.

## Usage

### Data Preparation

1. Ensure the dataset is structured as described above.
2. Update the `data_dir` variable in the code with the path to your dataset.

### Training the Model

```python
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Load Dataset
data_dir = '/path/to/your/dataset'  # Replace with the actual path to your dataset

batch_size = 32
img_height = 150
img_width = 150

train_ds = image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode='categorical'
)

val_ds = image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode='categorical'
)

# Define and Compile the Model
class_names = train_ds.class_names
num_classes = len(class_names)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the Model
early_stopping_cb = EarlyStopping(patience=5, restore_best_weights=True)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    callbacks=[early_stopping_cb]
)

# Evaluate the Model
test_loss, test_acc = model.evaluate(val_ds)
print(f"Test accuracy: {test_acc}")

# Visualize Training Results
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
Use code with caution.
Contributions
Contributions to enhance the model's accuracy, efficiency, or functionality are welcome. Potential areas for improvement include:
Expanding the dataset
Experimenting with different CNN architectures
Incorporating data augmentation techniques
Enhancing model evaluation metrics
