"""
real-fake-voice-detector/train.py

Cleaned and generalized version of the user's script.
Usage (example):
    python train.py --real_dir "/path/to/REAL" --fake_dir "/path/to/FAKE" --output_dir "./output" --epochs 20

Notes:
- This script does not include the Kaggle dataset. Provide local paths to the "REAL" and "FAKE" folders.
- Requires: numpy, librosa, pydub, scikit-learn, tensorflow, matplotlib, pandas (optional)
"""

import os
import argparse
import numpy as np
import librosa
from pydub import AudioSegment
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.callbacks import ModelCheckpoint

def extract_mfcc(audio_file, sr_target=None, max_length=20):
    """Load audio with librosa and extract (and pad/trim) MFCC features."""
    y, sr = librosa.load(audio_file, sr=sr_target)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    # pad or trim to fixed number of frames (max_length)
    if mfcc.shape[1] < max_length:
        padded = np.pad(mfcc, ((0,0),(0,max_length-mfcc.shape[1])), mode='constant')
    else:
        padded = mfcc[:, :max_length]
    return padded

def normalize_audio_file(audio_file, target_dBFS=-20.0):
    """Return AudioSegment normalized to target dBFS (used if needed)."""
    audio = AudioSegment.from_file(audio_file)
    change_in_dBFS = target_dBFS - audio.dBFS
    return audio.apply_gain(change_in_dBFS)

def load_features_from_folder(folder, max_files=None, max_length=20):
    """Load audio files from folder and extract MFCC features for each file."""
    features = []
    filenames = [f for f in os.listdir(folder) if f.lower().endswith(('.wav','.mp3','.flac','.ogg'))]
    if max_files:
        filenames = filenames[:max_files]
    for fname in filenames:
        path = os.path.join(folder, fname)
        try:
            # optionally normalize (commented out to avoid modifying files)
            # normalized = normalize_audio_file(path)
            mfcc = extract_mfcc(path, max_length=max_length)
            features.append(mfcc)
        except Exception as e:
            print(f"Warning: failed to process {path}: {e}")
    return np.array(features)

def build_cnn(input_shape, units=1024, kernel_size=3):
    model = Sequential([
        Conv1D(units, kernel_size, activation='relu', input_shape=input_shape),
        MaxPooling1D(2, padding='same'),
        Conv1D(units, kernel_size, activation='relu'),
        MaxPooling1D(2, padding='same'),
        Conv1D(units, kernel_size, activation='relu'),
        MaxPooling1D(2, padding='same'),
        Flatten(),
        Dense(units, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def plot_history(history, out_path):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,6))
    plt.subplot(2,1,1)
    plt.plot(history.history['accuracy'], label='train_acc')
    plt.plot(history.history.get('val_accuracy', []), label='val_acc')
    plt.title('Accuracy')
    plt.legend()
    plt.subplot(2,1,2)
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history.get('val_loss', []), label='val_loss')
    plt.title('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def main(args):
    real_dir = args.real_dir
    fake_dir = args.fake_dir
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    print("Loading real features...")
    real_features = load_features_from_folder(real_dir, max_files=args.max_files, max_length=args.max_length)
    print("Loading fake features...")
    fake_features = load_features_from_folder(fake_dir, max_files=args.max_files, max_length=args.max_length)

    if real_features.size == 0 or fake_features.size == 0:
        raise RuntimeError("No features loaded. Check your paths and audio files.")

    real_labels = np.ones(real_features.shape[0])
    fake_labels = np.zeros(fake_features.shape[0])

    X = np.concatenate((real_features, fake_features), axis=0)
    y = np.concatenate((real_labels, fake_labels), axis=0)

    # Shuffle and split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=42, stratify=y)

    # Ensure proper shape for Conv1D: (samples, timesteps, features)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2])
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])

    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)

    model = build_cnn(input_shape=(X_train.shape[1], X_train.shape[2]), units=args.units, kernel_size=args.kernel_size)
    model.summary()

    checkpoint_path = os.path.join(out_dir, "best_model.h5")
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

    history = model.fit(X_train, y_train, epochs=args.epochs, batch_size=args.batch_size,
                        validation_data=(X_test, y_test), callbacks=[checkpoint])

    # Save final weights
    model.save_weights(os.path.join(out_dir, "model_weights.h5"))

    # Plot training history
    plot_history(history, os.path.join(out_dir, "training_history.png"))

    # Evaluate
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {acc:.4f}, loss: {loss:.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_dir", required=True, help="Path to folder with real audio files (one folder)")
    parser.add_argument("--fake_dir", required=True, help="Path to folder with fake audio files (one folder)")
    parser.add_argument("--output_dir", default="./output", help="Folder to save models and plots")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=20)
    parser.add_argument("--units", type=int, default=1024)
    parser.add_argument("--kernel_size", type=int, default=3)
    args = parser.parse_args()
    main(args)
