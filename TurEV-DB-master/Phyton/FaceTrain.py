import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import logging

# Ensure the script uses UTF-8 encoding for stdout
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

# Configure logging to handle UTF-8 encoding
logging.basicConfig(filename='training.log', filemode='w', encoding='utf-8', level=logging.INFO)
logger = logging.getLogger()

# Define paths
dataset_path = r'C:\Users\berkh\Downloads\TurEV-DB-master\face'

# Image data generator
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Train and validation generators
train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(48, 48),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(48, 48),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Define the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')
])

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
try:
    model.fit(train_generator, epochs=25, validation_data=validation_generator)
    logger.info('Training completed successfully.')
    print('Training completed successfully.')
except UnicodeEncodeError as e:
    error_msg = f"UnicodeEncodeError: {e}"
    logger.error(error_msg)
    print(error_msg.encode('utf-8').decode('utf-8'))
except Exception as e:
    error_msg = f"An error occurred: {e}"
    logger.error(error_msg)
    print(error_msg.encode('utf-8').decode('utf-8'))

# Save the model
try:
    model.save('emotion_detection_model.h5')
    logger.info('Model saved successfully.')
    print('Model saved successfully.')
except UnicodeEncodeError as e:
    error_msg = f"UnicodeEncodeError: {e}"
    logger.error(error_msg)
    print(error_msg.encode('utf-8').decode('utf-8'))
except Exception as e:
    error_msg = f"An error occurred: {e}"
    logger.error(error_msg)
    print(error_msg.encode('utf-8').decode('utf-8'))
