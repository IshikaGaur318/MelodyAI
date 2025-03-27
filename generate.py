import numpy as np
import pickle
from tensorflow.keras.models import load_model
from music21 import stream, note, chord

# Load trained model
model = load_model("models/music_generator.h5")

# Load notes
with open("data/notes.pkl", "rb") as f:
    notes = pickle.load(f)

# Create mappings
unique_notes = sorted(set(notes))  # Ensure we store actual note names
note_to_int = {note: i for i, note in enumerate(unique_notes)}
int_to_note = {i: note for note, i in note_to_int.items()}

# Seed sequence
sequence_length = 100
start_sequence = notes[:sequence_length]
sequence = [note_to_int[n] for n in start_sequence]

# Generate music
output_notes = []
num_generated = 200  # Number of notes to generate

for _ in range(num_generated):
    # Prepare input
    X = np.reshape(sequence, (1, len(sequence), 1)) / float(len(unique_notes))

    # Predict next note probabilities
    prediction = model.predict(X, verbose=0)[0]

    # Use probability sampling for variety
    index = np.random.choice(len(prediction), p=prediction)

    # Convert back to note
    result_note = int_to_note[index]
    output_notes.append(result_note)

    # Update sequence
    sequence.append(index)
    sequence = sequence[1:]

# Convert output to MIDI
output_stream = stream.Stream()

for pattern in output_notes:
    if "." in pattern:  # Handle chords
        chord_notes = []
        for n in pattern.split("."):
            try:
                chord_notes.append(note.Note(n))  # Ensure valid note conversion
            except:
                print(f"Skipping invalid note: {n}")
        new_chord = chord.Chord(chord_notes)
        output_stream.append(new_chord)
    else:  # Single note
        try:
            output_stream.append(note.Note(pattern))  # Ensure valid note conversion
        except:
            print(f"Skipping invalid note: {pattern}")

# Save MIDI file
output_stream.write("midi", fp="generated_music.mid")

print("Generated MIDI saved as 'generated_music.mid'")
