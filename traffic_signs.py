import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten
from keras.models import Sequential, load_model

IMG_WIDTH = 30
IMG_HEIGHT = 30

"""
---------------Data Preprocessing---------------
"""

# Features are the Images, labels are the group (0-42) they belong to
features = []
labels = []
classes = 43

# Create list of images and their labels
for label in range(classes):
    subdir = os.path.join('Data', 'Train', str(label))
    for filename in os.listdir(subdir):
        print(filename)
        path = os.path.join(subdir, filename)

        image = Image.open(path)
        image = image.resize((IMG_WIDTH, IMG_HEIGHT))
        image = np.array(image)
        features.append(image)
        labels.append(label)

# Convert lists into numpy arrays
np.array(features)
np.array(labels)

# See shape of data
print("Shape of Data")
print(f"Features: {features.shape}, labels: {labels.shape}", end='\n\n\n')

# Split data into train and test sets
# Train on 80% of the data
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Convert labels with one hot encoding
y_train = to_categorical(y_train, classes)
y_test = to_categorical(y_test, classes)


"""
--------------Building and Training the Model---------------
"""

# Dropout used to prevent overfitting
model = Sequential()
# Input layer, need to state the shape of the input
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)))
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))
# Output layer with 43 nodes, one for each class
model.add(Dense(43, activation='softmax'))
# Compilation of the model
# categorical_crossentropy used as we have categorical data
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# 15 passes through the training data
# batch_size is 64, 64 features analysed before weights updated
epochs = 15
history = model.fit(X_train, y_train, batch_size=64, epochs=epochs, validation_data=(X_test, y_test))
model.save("my_model.h5")