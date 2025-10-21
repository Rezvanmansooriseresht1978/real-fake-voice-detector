# real-fake-voice-detector(Deep Learning, CNN, MFCC)

**Author:** Rezvan Mansoori

This project implements a deep learning-based system to automatically distinguish real human voices from AI-generated (deepfake) voices. It combines audio signal processing and Convolutional Neural Networks (CNNs) to achieve up to 84% accuracy in classification.

*** Overview

In recent years, deepfake audio has become an emerging challenge in cybersecurity, media integrity, and online communication.
This project aims to develop a reliable audio classification model that identifies whether a given voice sample is genuine or synthetic.
By extracting MFCC (Mel-Frequency Cepstral Coefficients) from audio signals and training a 1D CNN, this system learns to capture spectral patterns unique to human and AI-generated voices.

*** Background & Motivation

Fake voice generation techniques are advancing rapidly, making it increasingly difficult to detect audio forgeries by ear alone.
This project was motivated by the question:
“Can we build an automated model that can learn to detect deepfake voices using signal-level features?”
The approach integrates traditional signal processing (MFCC extraction, normalization) with modern deep learning architectures (Conv1D layers).

*** Technical Details

Component Description 
Language Python 3 
Frameworks TensorFlow / Keras, scikit-learn 
Libraries NumPy, Pandas, Matplotlib, Librosa, pydub 
Model Type 1D Convolutional Neural Network (CNN) 
Feature Extraction MFCC (20 coefficients per frame) 
Dataset Split 80% Training / 20% Testing 
Evaluation Metrics Accuracy, Confusion Matrix, Classification Report

*** Model Architecture & Methodology

Audio Preprocessing:
Normalized all input signals using pydub for consistent volume levels.
Extracted MFCC features from each file with Librosa.

Feature Handling:
Padded or trimmed each MFCC to a uniform length.
Labeled real voices as 1 and fake voices as 0.

Model Design:
Three stacked Conv1D layers with ReLU activation.
MaxPooling after each convolution.
Dense layers for final classification with Sigmoid activation.

Training & Tuning:
Used Adam optimizer and Binary Cross-Entropy loss.
Experimented with hyperparameter tuning via GridSearchCV.

*** Dataset

Fake-or-Real (FoR) Audio Dataset
Format: .wav and .mp3 audio files
Structure: 
AUDIO
    REAL
    FAKE
    
*** Results
Training Accuracy: ~84%
Validation Accuracy: ~81%
Loss Trend: Steady convergence after ~15 epochs
Confusion Matrix: Demonstrated strong true positive and negative balance
These results confirm that MFCC features combined with a CNN can effectively capture discriminative characteristics between genuine and fake audio.

*** How to Run
pip install -r requirements.txt
python src/train.py\ 
--real_dir "E:/Datasets/FoR/REAL" \ 
--fake_dir "E:/Datasets/FoR/FAKE" \ 
--output_dir "./output" \ 
--epochs 20

*** Skills Demonstrated
Deep Learning: CNN design, tuning, and evaluation
Signal Processing: MFCC feature extraction and normalization
Python Development: Modular, clean code with documentation
Data Handling: Preprocessing, splitting, and feature alignment
Research Thinking: Designing a model to solve a real-world AI challenge

*** References
Kaggle Dataset: Fake-or-Real (FoR) Audio Dataset 
Keras API Documentation 
Librosa Audio Feature Extraction Guide

## Author
Rezvan Mansoori
M.A. in Software Engineering
Focused on AI, Deep Learning, and Audio Recognition

If you find this project useful or interesting, feel free to star  the repository and explore my other AI and Python projects!
