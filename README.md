# Emotion Recognition from Audio Using TurEV-DB

This repository contains a script to record audio, extract MFCC features, and predict the emotion of the recorded audio using a pre-trained Convolutional Neural Network (CNN) model, leveraging the Turkish Emotion Voice Database (TurEV-DB).

## Prerequisites

Before running the script, ensure you have the following installed:

- Python 3.7 or later
- `pip` (Python package installer)

## Installation

1. Clone the repository to your local machine:
    ```bash
    git clone https://github.com/berkhantiryakiler/emotion-recognition-audio.git
    cd emotion-recognition-audio
    ```

2. Install the required Python packages:
    ```bash
    pip install numpy sounddevice librosa tensorflow soundfile
    ```

3. Ensure you have the pre-trained model file (`emotion_cnn_model.h5`) and the label classes file (`label_classes.npy`) in the repository directory. You can download these files from the links provided or place your own trained model and labels file in the directory.

## Dataset

This project uses the Turkish Emotion Voice Database (TurEV-DB). For more details about the dataset, visit the [TurEV-DB GitHub page](https://github.com/Xeonen/TurEV-DB).

## Running the Script

To run the script, use the following command:
```bash
VoiceUs.py
