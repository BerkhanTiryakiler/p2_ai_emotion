import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

main_folder_path = r'C:\Users\berkh\Downloads\TurEV-DB-master\TurEV-DB-master\Sound Source'

X = []
y = []

label_dict = {'angry': 0, 'sad': 1, 'happy': 2, 'calm': 3}

def pad_truncate_sequence(seq, max_len):
    if seq.shape[1] > max_len:
        return seq[:, :max_len]
    else:
        return np.pad(seq, ((0, 0), (0, max_len - seq.shape[1])), 'constant')

fixed_length = 173  

for label_name, label in label_dict.items():
    folder_path = os.path.join(main_folder_path, label_name)
    if os.path.isdir(folder_path):
        for filename in os.listdir(folder_path):
            if filename.endswith(".wav"): 
                file_path = os.path.join(folder_path, filename)
                
                try:
                    audio, sr = librosa.load(file_path, sr=None)
                except Exception as e:
                    try:
                        print(f"Error loading {file_path}: {e}")
                    except UnicodeEncodeError:
                        print("Error loading file: Encoding issue.")
                    continue
                
                try:
                    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
                    mfccs = (mfccs - np.mean(mfccs)) / np.std(mfccs)  
                    
                    mfccs = pad_truncate_sequence(mfccs, fixed_length)
                    
                    mfccs = mfccs.flatten()
                    
                    X.append(mfccs)
                    y.append(label)
                except Exception as e:
                    try:
                        print(f"Error processing {file_path}: {e}")
                    except UnicodeEncodeError:
                        print("Error processing file: Encoding issue.")
                    continue

if not X or not y:
    print("No data collected. Please check the paths and data format.")
else:
    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Sequential()
    model.add(Dense(256, input_shape=(fixed_length * 13,), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(len(label_dict), activation='softmax')) 

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test Accuracy: {accuracy * 100:.2f}%')

    model.save('audio_classification_model.h5')

    loaded_model = tf.keras.models.load_model('audio_classification_model.h5')
