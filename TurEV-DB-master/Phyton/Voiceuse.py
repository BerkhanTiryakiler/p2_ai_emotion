import numpy as np
import sounddevice as sd
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
import sys
import io
import time

# Adjust stdout encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Load the trained model
model = load_model('audio_classification_model2.h5')

# Define parameters
duration = 3  # Duration of recording in seconds
sample_rate = 22050  # Sample rate for recording
fixed_length = 173  # Length of the MFCC feature sequences
interval = 5  # Interval between predictions in seconds

label_dict = {0: 'angry', 1: 'sad', 2: 'happy', 3: 'calm'}

def pad_truncate_sequence(seq, max_len):
    if seq.shape[1] > max_len:
        return seq[:, :max_len]
    else:
        return np.pad(seq, ((0, 0), (0, max_len - seq.shape[1])), 'constant')

def get_mfccs(audio, sample_rate):
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
    mfccs = (mfccs - np.mean(mfccs)) / np.std(mfccs)
    mfccs = pad_truncate_sequence(mfccs, fixed_length)
    mfccs = mfccs.flatten()
    return mfccs

def record_audio(duration, sample_rate):
    print("Recording...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()  # Wait until the recording is finished
    audio = audio.flatten()
    print("Recording complete.")
    return audio

def predict_emotion(audio, sample_rate):
    mfccs = get_mfccs(audio, sample_rate)
    mfccs = np.expand_dims(mfccs, axis=0)  # Add batch dimension
    predictions = model.predict(mfccs)
    predicted_label = np.argmax(predictions)
    return label_dict[predicted_label]

def main():
    try:
        while True:
            # Record audio from the microphone
            audio = record_audio(duration, sample_rate)

            # Predict the emotion
            emotion = predict_emotion(audio, sample_rate)
            print(f'Predicted Emotion: {emotion}')

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
