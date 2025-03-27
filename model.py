import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def build_model(input_shape, num_classes=3158):  # Ensure num_classes is passed
    model = Sequential([
        LSTM(256, return_sequences=True, input_shape=input_shape),
        LSTM(256),
        Dense(256, activation='relu'),
        Dense(num_classes, activation='softmax')  # Ensure num_classes is used
    ])
    return model
