import os
import sys
import io
import cv2
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import pyaudio
import threading
import matplotlib.pyplot as plt
from librosa.feature import melspectrogram
from librosa.display import specshow
import speech_recognition as sr

# Set UTF-8 encoding for stdout
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Load the pre-trained models
face_model_path = 'emotion_detection_model.h5'
voice_model_path = 'audio_classification_model2.h5'

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

# Function to preprocess face image
def preprocess_face_image(face):
    face = cv2.resize(face, (48, 48))
    face = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)  # Convert grayscale to RGB
    face = face.astype('float32') / 255.0
    face = img_to_array(face)
    face = np.expand_dims(face, axis=0)
    return face

# Function to preprocess audio chunk into spectrogram
def preprocess_audio_chunk(audio_chunk, sr=22050, n_mels=128, fixed_length=173):
    spectrogram = melspectrogram(y=audio_chunk, sr=sr, n_mels=n_mels)
    spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    if spectrogram.shape[1] > fixed_length:
        spectrogram = spectrogram[:, :fixed_length]
    else:
        spectrogram = np.pad(spectrogram, ((0, 0), (0, fixed_length - spectrogram.shape[1])), 'constant')
    spectrogram = spectrogram.flatten()
    spectrogram = np.expand_dims(spectrogram, axis=0)
    return spectrogram

# Function to capture audio in chunks
def capture_audio(stream, chunk_size=1024):
    global audio_frames, capturing_audio
    audio_frames = []
    while capturing_audio:
        data = stream.read(chunk_size, exception_on_overflow=False)
        audio_frames.append(np.frombuffer(data, dtype=np.int16))  # Changed dtype to match PyAudio format

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize speech recognizer for ambient noise adjustment
r = sr.Recognizer()
mic = sr.Microphone()

# Adjust for ambient noise
with mic as source:
    print("Calibrating for ambient noise...")
    r.adjust_for_ambient_noise(source, duration=5)
    print("Calibration completed.")

# Initialize microphone
audio = pyaudio.PyAudio()
stream = audio.open(format=pyaudio.paInt16,  # Changed format
                    channels=1,
                    rate=22050,
                    input=True,
                    frames_per_buffer=1024)

capturing_audio = True
audio_frames = []
audio_thread = threading.Thread(target=capture_audio, args=(stream,))
audio_thread.start()

voice_emotion = "xxxx"
recognized_text = ""

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

        # Capture audio and predict emotion every few seconds
        if len(audio_frames) >= 22050:  # Process every second of audio
            print("Processing audio chunk...")  # Debug log
            audio_chunk = np.concatenate(audio_frames[:22050])
            audio_frames = audio_frames[22050:]
            try:
                # Convert audio to float32 for librosa processing
                audio_chunk = audio_chunk.astype(np.float32) / np.iinfo(np.int16).max

                # Plot the spectrogram for debugging purposes
                plt.figure(figsize=(10, 4))
                specshow(librosa.power_to_db(melspectrogram(y=audio_chunk, sr=22050, n_mels=128), ref=np.max), sr=22050, x_axis='time', y_axis='mel')
                plt.colorbar(format='%+2.0f dB')
                plt.title("Mel Spectrogram")
                plt.show()

                audio_features = preprocess_audio_chunk(audio_chunk)
                print(f"Audio features shape: {audio_features.shape}")  # Debug log
                print(f"Audio features: {audio_features}")  # Debug log

                voice_prediction = voice_model.predict(audio_features)
                print(f"Voice prediction: {voice_prediction}")  # Debug log

                voice_emotion = voice_labels[np.argmax(voice_prediction)]
                print(f"Detected voice emotion: {voice_emotion}")  # Debug log

                # Recognize speech using speech_recognition
                with sr.AudioFile(io.BytesIO(audio_chunk.astype(np.int16).tobytes())) as source:
                    r.adjust_for_ambient_noise(source)
                    audio_data = r.record(source)
                    try:
                        recognized_text = r.recognize_google(audio_data)
                        print(f"Recognized Text: {recognized_text}")
                    except sr.UnknownValueError:
                        print("Google Speech Recognition could not understand audio")
                    except sr.RequestError as e:
                        print(f"Could not request results from Google Speech Recognition service; {e}")

            except Exception as e:
                print(f"Error during audio processing: {e}")

        # Create a black strip below the frame to display voice emotion and recognized text
        height, width, _ = frame.shape
        black_strip = np.zeros((100, width, 3), dtype=np.uint8)
        cv2.putText(black_strip, f'Voice Emotion: {voice_emotion}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(black_strip, f'Recognized Text: {recognized_text}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        frame_with_strip = np.vstack((frame, black_strip))

        # Display the video frame with face and voice emotion labels
        cv2.imshow('Emotion Detection', frame_with_strip)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Release resources
    capturing_audio = False
    audio_thread.join()
    cap.release()
    stream.stop_stream()
    stream.close()
    audio.terminate()
    cv2.destroyAllWindows()
