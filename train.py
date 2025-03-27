import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from model import build_model
from tqdm import tqdm
import sys

# Load notes dataset
with open("data/notes.pkl", "rb") as f:
    notes = pickle.load(f)

# Mapping unique notes to integer indices
unique_notes = sorted(set(notes))
note_to_int = {note: i for i, note in enumerate(unique_notes)}
num_classes = len(unique_notes)  # Must match model's output layer

# Convert notes into input-output sequences
sequence_length = 100
num_samples = len(notes) - sequence_length

input_data = np.zeros((num_samples, sequence_length), dtype=np.int32)
output_data = np.zeros(num_samples, dtype=np.int32)  # Labels must be integers

print(f"Preparing dataset with {num_samples} sequences...")

for i in tqdm(range(num_samples), desc="Processing Sequences", unit="seq"):
    input_data[i] = [note_to_int[n] for n in notes[i:i + sequence_length]]
    output_data[i] = note_to_int[notes[i + sequence_length]]

# Debugging: Ensure output_data labels are in range
print(f"Label range: min={output_data.min()}, max={output_data.max()}, num_classes={num_classes}")

# Check label validity
assert output_data.min() >= 0, "Error: Found negative labels!"
assert output_data.max() < num_classes, f"Error: Labels exceed num_classes ({output_data.max()} >= {num_classes})!"

# Normalize input
X = np.reshape(input_data, (num_samples, sequence_length, 1)) / float(num_classes)

# Convert to TensorFlow dataset
batch_size = 512
train_dataset = tf.data.Dataset.from_tensor_slices((X, output_data)).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

# Build the model
model = build_model((sequence_length, 1), num_classes=num_classes)

# Check model output shape
if model.output_shape[-1] != num_classes:
    raise ValueError(f"Model output shape {model.output_shape} does not match num_classes={num_classes}!")

# Compile model
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

try:
    print("Starting Training...")
    history = model.fit(
    train_dataset, 
    epochs=50, 
    steps_per_epoch=500,  # Limits steps per epoch to speed up training
    verbose=1, 
    callbacks=[
        tf.keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: print(f"Epoch {epoch + 1}/50 - Loss: {logs['loss']:.4f}, Acc: {logs['accuracy']:.4f}")
        ),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5, patience=3, verbose=1)  # Reduce LR if no improvement
    ]
)


    # Save trained model
    model.save("models/music_generator.h5")
    print("Training complete. Model saved!")

except KeyboardInterrupt:
    print("\nTraining interrupted! Saving current model...")
    model.save("models/music_generator_interrupted.h5")
    print("Model saved. Exiting.")
    sys.exit(1)
