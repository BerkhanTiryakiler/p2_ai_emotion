import os
import sys
import io
import cv2
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import pyaudio
import threading
import sounddevice as sd
import time

# Set UTF-8 encoding for stdout
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Load the pre-trained models
face_model_path = 'emotion_detection_model.h5'
voice_model_path = 'audio_classification_model.h5'

# Load models with error handling
try:
    face_model = load_model(face_model_path)
    print("Face model loaded successfully.")
except Exception as e:
    print(f"Error loading face model: {e}")

try:
    voice_model = load_model(voice_model_path)
    print("Voice model loaded successfully.")
except Exception as e:
    print(f"Error loading voice model: {e}")

# Dictionary to map labels
face_labels = {0: 'angry', 1: 'sad', 2: 'happy', 3: 'calm'}
voice_labels = {0: 'angry', 1: 'sad', 2: 'happy', 3: 'calm'}

# Parameters for voice detection
duration = 3  # Duration of recording in seconds
sample_rate = 22050  # Sample rate for recording
fixed_length = 173  # Length of the MFCC feature sequences
interval = 5  # Interval between predictions in seconds

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
    predictions = voice_model.predict(mfccs)
    predicted_label = np.argmax(predictions)
    return voice_labels[predicted_label]

# Function to preprocess face image
def preprocess_face_image(face):
    face = cv2.resize(face, (48, 48))
    face = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)  # Convert grayscale to RGB
    face = face.astype('float32') / 255.0
    face = img_to_array(face)
    face = np.expand_dims(face, axis=0)
    return face

# Initialize webcam
cap = cv2.VideoCapture(0)

voice_emotion = "xxxx"

def voice_detection():
    global voice_emotion
    try:
        while True:
            # Record audio from the microphone
            audio = record_audio(duration, sample_rate)

            # Predict the emotion
            emotion = predict_emotion(audio, sample_rate)
            print(f'Predicted Voice Emotion: {emotion}')
            voice_emotion = emotion

            # Wait for the next interval
            time.sleep(interval)
    except KeyboardInterrupt:
        print("Voice detection interrupted by user.")
    except Exception as e:
        print(f"An error occurred in voice detection: {e}")

# Start the voice detection in a separate thread
voice_thread = threading.Thread(target=voice_detection)
voice_thread.start()

try:
    while True:
        # Capture video frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]
            face_image = preprocess_face_image(face)
            face_prediction = face_model.predict(face_image)
            face_label = face_labels[np.argmax(face_prediction)]

            # Draw rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, face_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        # Create a black strip below the frame to display voice emotion
        height, width, _ = frame.shape
        black_strip = np.zeros((50, width, 3), dtype=np.uint8)
        cv2.putText(black_strip, f'Voice Emotion: {voice_emotion}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        frame_with_strip = np.vstack((frame, black_strip))

        # Display the video frame with face and voice emotion labels
        cv2.imshow('Emotion Detection', frame_with_strip)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    # Ensure the voice detection thread stops
    voice_thread.join()
