# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras.datasets import imdb  # Load IMDB dataset
from tensorflow.keras.preprocessing.sequence import pad_sequences  # Padding sequences
from tensorflow.keras.models import Sequential  # Sequential model
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Dropout  # Layers for RNN
from tensorflow.keras.callbacks import EarlyStopping  # Early stopping callback

from sklearn.metrics import classification_report, roc_curve, auc  # Performance metrics

import keras_tuner as kt  # Hyperparameter tuning
from keras_tuner.tuners import RandomSearch  # Random search tuner

import warnings
warnings.filterwarnings("ignore")  # Ignore warnings

# Load IMDB dataset (50,000 reviews: 0=negative, 1=positive)
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)  # Use top 10,000 words

# Data preprocessing
# Pad sequences to ensure all reviews have the same length
maxlen = 100
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)

# Function to build the model with hyperparameters
def build_model(hp):  # hp: hyperparameters
    model = Sequential()  # Base model

    # Embedding layer: Convert words to vectors
    model.add(Embedding(input_dim=10000,
                        output_dim=hp.Int("embedding_output", min_value=32, max_value=128, step=32),  # Vector dimensions
                        input_length=maxlen))
    # SimpleRNN layer: Process sequential data
    model.add(SimpleRNN(units=hp.Int("rnn_units", min_value=32, max_value=128, step=32)))
    # Dropout layer: Prevent overfitting by randomly dropping neurons
    model.add(Dropout(rate=hp.Float("dropout_rate", min_value=0.2, max_value=0.5, step=0.1)))
    # Output layer: 1 neuron with sigmoid activation for binary classification
    model.add(Dense(1, activation="sigmoid"))

    # Compile the model
    model.compile(optimizer=hp.Choice("optimizer", ["adam", "rmsprop"]),  # Optimizer choice
                  loss="binary_crossentropy",  # Binary cross-entropy loss
                  metrics=["accuracy", "AUC"])  # Metrics: accuracy and AUC
    return model

# Hyperparameter search: Random Search
tuner = RandomSearch(
    build_model,  # Function to build the model
    objective="val_loss",  # Minimize validation loss
    max_trials=2,  # Try 2 different models
    executions_per_trial=1,  # Execute each trial once
    directory="rnn_tuner_directory",  # Directory to save models
    project_name="imdb_rnn"  # Project name
)

# Train the model with early stopping
early_stopping = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

tuner.search(x_train, y_train,
             epochs=5,
             validation_split=0.2,
             callbacks=[early_stopping]
             )

# Get the best model
best_model = tuner.get_best_models(num_models=1)[0]  # Best performing model

# Evaluate the best model on the test set
loss, accuracy, auc_score = best_model.evaluate(x_test, y_test)
print(f"Test loss: {loss:.3f}, Test accuracy: {accuracy:.3f}, Test AUC: {auc_score:.3f}")

# Make predictions and evaluate model performance
y_pred_prob = best_model.predict(x_test)
y_pred = (y_pred_prob > 0.5).astype("int32")

print(classification_report(y_test, y_pred))

# Calculate ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)  # False positive rate and true positive rate
roc_auc = auc(fpr, tpr)  # Area under the ROC curve

# Visualize ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC Curve (area=%0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color="blue", lw=2, linestyle="--")  # Random prediction line
plt.xlim([0, 1])
plt.ylim([0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Example')
plt.legend()
plt.show()