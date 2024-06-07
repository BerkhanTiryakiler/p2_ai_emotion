import numpy as np
import sounddevice as sd
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
import soundfile as sf  # Import soundfile for saving audio
import sys
import io
import time
import os

# Adjust stdout encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Load the trained model
model = load_model('emotion_cnn_model.h5')
label_classes = np.load('label_classes.npy', allow_pickle=True)

# Define parameters
duration = 3  # Duration of recording in seconds
sample_rate = 22050  # Sample rate for recording
interval = 5  # Interval between predictions in seconds
save_dir = 'recordings'  # Directory to save recordings and MFCCs

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

label_dict = {i: label for i, label in enumerate(label_classes)}

def get_mfccs(audio, sample_rate):
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs = np.mean(mfccs.T, axis=0)  # Take the mean over time
    return mfccs

def record_audio(duration, sample_rate):
    print("Recording...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()  # Wait until the recording is finished
    audio = audio.flatten()
    print("Recording complete.")
    return audio

def save_recording(audio, mfccs, index):
    audio_path = os.path.join(save_dir, f'audio_{index}.wav')
    mfcc_path = os.path.join(save_dir, f'mfcc_{index}.npy')
    sf.write(audio_path, audio, sample_rate)  # Use soundfile to save audio
    np.save(mfcc_path, mfccs)
    print(f"Saved audio to {audio_path} and MFCCs to {mfcc_path}")

def predict_emotion(audio, sample_rate, index):
    mfccs = get_mfccs(audio, sample_rate)
    save_recording(audio, mfccs, index)
    mfccs = np.expand_dims(mfccs, axis=0)  # Add batch dimension
    mfccs = np.expand_dims(mfccs, axis=-1)  # Add channel dimension
    predictions = model.predict(mfccs)
    predicted_label = np.argmax(predictions)
    return label_dict[predicted_label]

def main():
    index = 0
    try:
        while True:
            # Record audio from the microphone
            audio = record_audio(duration, sample_rate)

            # Predict the emotion
            emotion = predict_emotion(audio, sample_rate, index)
            print(f'Predicted Emotion: {emotion}')

            # Increment index for the next recording
            index += 1

            # Wait for the next interval
            time.sleep(interval)
    except UnicodeEncodeError as e:
        print(f"UnicodeEncodeError: {e}")
    except KeyboardInterrupt:
        print("Process interrupted by user.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
