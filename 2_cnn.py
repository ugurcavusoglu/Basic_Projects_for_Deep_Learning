# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import cifar10  # Load CIFAR-10 dataset
from tensorflow.keras.utils import to_categorical  # Convert labels to one-hot encoding
from tensorflow.keras.models import Sequential  # Sequential model
from tensorflow.keras.layers import Dense, Dropout, Flatten  # Layers for classification
from tensorflow.keras.layers import Conv2D, MaxPooling2D  # Layers for feature extraction
from tensorflow.keras.optimizers import RMSprop  # Optimizer
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Data augmentation

from sklearn.metrics import classification_report  # Detailed performance report

import warnings
warnings.filterwarnings("ignore")  # Ignore warnings

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Visualization
class_labels = ["Airplane", "Automobile", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]

# Visualize some images and their labels
fig, axes = plt.subplots(1, 5, figsize=(15, 10))
for i in range(5):
    axes[i].imshow(x_train[i])
    label = class_labels[int(y_train[i])]
    axes[i].set_title(label)
    axes[i].axis("off")

# Data normalization
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# One-hot encoding for labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,  # Rotate images by up to 20 degrees
    width_shift_range=0.2,  # Shift images horizontally by up to 20%
    height_shift_range=0.2,  # Shift images vertically by up to 20%
    shear_range=0.2,  # Apply shear transformation
    zoom_range=0.2,  # Zoom in/out by up to 20%
    horizontal_flip=True,  # Flip images horizontally
    fill_mode="nearest"  # Fill missing pixels with the nearest value
)

datagen.fit(x_train)

# Create CNN model (base model)
model = Sequential()

# Feature Extraction: CONV => RELU => CONV => RELU => POOL => DROPOUT
model.add(Conv2D(32, (3, 3), padding="same", activation="relu", input_shape=x_train.shape[1:]))
model.add(Conv2D(32, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Feature Extraction: CONV => RELU => CONV => RELU => POOL => DROPOUT
model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Classification: FLATTEN, DENSE, RELU, DROPOUT, DENSE (OUTPUT LAYER)
model.add(Flatten())  # Convert 2D matrix to 1D vector
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation="softmax"))  # Output layer with 10 classes

# Print model summary
model.summary()

# Compile the model
model.compile(optimizer=RMSprop(learning_rate=0.0001, decay=1e-6),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# Train the model
history = model.fit(datagen.flow(x_train, y_train, batch_size=512),  # Data augmentation applied
                    epochs=20,
                    validation_data=(x_test, y_test))  # Validation set

# Make predictions on the test set
y_pred = model.predict(x_test)
y_pred_class = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Print classification report
report = classification_report(y_true, y_pred_class, target_names=class_labels)
print(report)

# Plot training and validation metrics
plt.figure()

# Loss curves
plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()

# Accuracy curves
plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()

plt.grid()
plt.tight_layout()
plt.show()