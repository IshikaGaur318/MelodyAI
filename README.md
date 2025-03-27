# 🎶 AI Music Generator

🚀 A deep learning-based music generation project that trains on MIDI files to generate original music compositions.

## 📌 Features
- Processes MIDI files into a structured dataset 🎼
- Trains an LSTM-based neural network for music generation 🧠
- Generates new music sequences and saves them as MIDI files 🎹

## 📂 Project Structure
```
├── data/               # MIDI dataset and processed note sequences
├── models/             # Trained models
├── src/
│   ├── dataset.py      # Prepares dataset from MIDI files
│   ├── model.py        # Defines the LSTM model architecture
│   ├── train.py        # Trains the model on processed data
│   ├── generate.py     # Generates music using the trained model
├── README.md           # Project documentation
```

## 📦 Installation
1️⃣ Clone the repository:
```bash
git clone https://github.com/yourusername/music-generation-ai.git
cd music-generation-ai
```

2️⃣ Install dependencies:
```bash
pip install -r requirements.txt
```

## 🎼 Dataset
Place MIDI files inside the `data/` directory before running `dataset.py`.

## 🏋️ Training
Run the following command to train the model:
```bash
python src/train.py
```

## 🎵 Music Generation
Generate new music using the trained model:
```bash
python src/generate.py
```

The generated music will be saved as `generated_music.mid`.

## 📌 Requirements
- Python 3.8+
- TensorFlow
- NumPy
- Music21
- tqdm

## ⭐ Contributing
Pull requests are welcome! Feel free to fork the repo and submit improvements. 😊

## 📜 License
MIT License © 2025 Your Name
