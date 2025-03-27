import glob
import pickle
import multiprocessing
from music21 import converter, note, chord
from tqdm import tqdm

def process_midi(file):
    try:
        midi = converter.parse(file)
        notes = []

        for element in midi.flatten().notes:  # Use .flatten() instead of .flat
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))
                
        return notes  # Return processed notes for this file
    except Exception as e:
        print(f"Error processing {file}: {e}")
        return []  # Return empty list if error occurs

def load_midi_files(dataset_path, num_workers=4):
    midi_files = glob.glob(dataset_path + "/*.midi")
    
    print(f"Found {len(midi_files)} MIDI files. Processing with {num_workers} workers...")

    notes = []
    
    try:
        with multiprocessing.Pool(num_workers) as pool:
            results = list(tqdm(pool.imap(process_midi, midi_files), total=len(midi_files), desc="Processing MIDI Files"))

        # Flatten the list of lists
        for result in results:
            notes.extend(result)

        # Save processed data
        with open("data/notes.pkl", "wb") as f:
            pickle.dump(notes, f)

    except KeyboardInterrupt:
        print("\nUser interrupted! Cleaning up processes...")
        pool.terminate()  # Terminate all processes
        pool.join()       # Wait for processes to exit
        print("Process terminated. Exiting.")
        exit(1)  # Exit with error code

    return notes

if __name__ == "__main__":
    notes = load_midi_files("data", num_workers=8)  # Adjust num_workers based on CPU cores
    print(f"Loaded {len(notes)} notes")
