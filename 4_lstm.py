# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_20newsgroups  # Load 20 Newsgroups dataset
from sklearn.preprocessing import LabelEncoder  # Convert labels to numerical format
from sklearn.model_selection import train_test_split  # Split data into train and test sets

from tensorflow.keras.preprocessing.text import Tokenizer  # Convert text data to sequences
from tensorflow.keras.preprocessing.sequence import pad_sequences  # Pad sequences to the same length
from tensorflow.keras.models import Sequential  # Sequential model
from tensorflow.keras.layers import Embedding, Dense, Dropout, LSTM  # Layers for LSTM model
from tensorflow.keras.callbacks import EarlyStopping  # Early stopping callback

import warnings
warnings.filterwarnings("ignore")  # Ignore warnings

# Load 20 Newsgroups dataset
newsgroup = fetch_20newsgroups(subset="all")  # Load both train and test sets
X = newsgroup.data  # X: Text data
y = newsgroup.target  # y: Labels

# Tokenize text data and apply padding
tokenizer = Tokenizer(num_words=10000)  # Use top 10,000 words
tokenizer.fit_on_texts(X)  # Fit tokenizer on text data
X_sequences = tokenizer.texts_to_sequences(X)  # Convert texts to sequences of numbers
X_padded = pad_sequences(X_sequences, maxlen=100)  # Pad sequences to max length of 100

# Convert labels to numerical format using LabelEncoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(X_padded, y_encoded, test_size=0.2, random_state=42)

# Function to build LSTM model
def build_lstm_model():
    model = Sequential()

    # Layers: Embedding + LSTM + Dropout + Output

    # Embedding layer: Convert words to vectors
    # input_dim: Total number of words
    # output_dim: Dimension of word vectors
    # input_length: Length of each input text
    model.add(Embedding(input_dim=10000, output_dim=64, input_length=100))

    # LSTM layer: Process sequential data
    # units: Number of LSTM cells
    # return_sequences: Return only the last output (not the full sequence)
    model.add(LSTM(units=64, return_sequences=False))

    # Dropout layer: Prevent overfitting by randomly dropping neurons
    model.add(Dropout(0.5))  # Dropout rate of 0.5

    # Output layer: 20 neurons with softmax activation for multi-class classification
    model.add(Dense(20, activation="softmax"))

    # Compile the model
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

# Build the LSTM model
model = build_lstm_model()
model.summary()  # Print model summary

# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_split=0.1,
                    callbacks=[early_stopping])

# Evaluate the model on the test set
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

# Plot training and validation metrics
plt.figure()

# Plot training and validation loss
plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid("True")

# Plot training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid("True")

plt.show()