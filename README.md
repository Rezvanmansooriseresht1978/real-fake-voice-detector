# real-fake-voice-detector

**Author:** Rezvan Mansoori

A Python project to classify *real* vs *fake* (deepfake) audio samples.
This repository documents and packages the original project. The dataset used was downloaded from Kaggle: **The Fake-or-Real (FoR) Dataset (deepfake audio)**. The dataset is **not included** in this repository; please download it from Kaggle and point the script to the `REAL` and `FAKE` folders on your machine.

## Contents
- `src/train.py` - Main training script (cleaned and generalized version of the original code)
- `requirements.txt` - Python dependencies
- `README.md` - This file
- `output/` - (generated when you run training) contains model weights and plots

## Quickstart

1. Clone the repository or download and extract zip.
2. Install dependencies (preferably in a virtual environment):

```bash
pip install -r requirements.txt
```

3. Download the dataset (Kaggle): **The Fake-or-Real (FoR) Dataset (deepfake audio)** and prepare two folders:
```
/path/to/REAL
/path/to/FAKE
```

4. Run training (example):

```bash
python src/train.py --real_dir "/path/to/REAL" --fake_dir "/path/to/FAKE" --output_dir "./output" --epochs 20
```

After training, `output/` will contain `best_model.h5`, `model_weights.h5`, and `training_history.png`.

## Approach (summary)

- Extract MFCC features from each audio file (padded/truncated to fixed length)
- Build a Conv1D-based neural network using TensorFlow / Keras
- Train the model using an 80/20 train-test split
- Save best model and plot training accuracy/loss

## Notes & Caveats

- Audio datasets are often large. This repo does **not** include the Kaggle dataset; link and citation must be followed on Kaggle to download the FoR dataset.
- Depending on the dataset and preprocessing, model performance may vary — reproducing the original 60%+ accuracy depends on the exact preprocessing, dataset split, and hyperparameters.
- You may need to install `ffmpeg` or `libav` for `pydub` and `librosa` to read certain audio formats.
- If you want an interactive notebook version, open `notebook/real_fake_voice_detector.ipynb` (included).

## Author
Rezvan Mansoori
