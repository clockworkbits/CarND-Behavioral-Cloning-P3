import csv
import cv2
import numpy as np
import random
import sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

# Load the data from the csv file
print("Loading the data from the csv file")

correction_factor = 0.2

samples_list = [] # Contains tuples (path, steering angle, flag if image should be flipped)
with open("data/driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        measurement = float(line[3])
        # Filter some of the near center samples
        if (measurement < 0.025 and measurement > -0.025 and random.choice([True, False])):
            continue
        
        # Iterate through Center, Left, Right images
        for i in range(3):
            source_path = line[i] # 0 index is for the center camera
            filename = source_path.split('/')[-1]
            current_path = 'data/IMG/' + filename

            # Add the measurement for the given image
            if i == 0: # Center
                adjusted_measurement = measurement
            elif i == 1: # Left
                adjusted_measurement = measurement + correction_factor
            else: # i == 2 - Right
                adjusted_measurement = measurement - correction_factor
            
            samples_list.append((current_path, adjusted_measurement, False)) # Boolean means if the image should be flipped
            samples_list.append((current_path, -adjusted_measurement, True))
            
print("Number of samples = {}".format(len(samples_list)))

def flip_image(image):
    """Helfper function to flip images horizontally"""
    return cv2.flip(image, flipCode=1)


def generator(samples, batch_size=32):
    """ Generator function for keras"""
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                path, angle, flip = batch_sample
                image = cv2.imread(path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # convert to RGB
                images.append(flip_image(image) if flip else image)
                angles.append(angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# Split the data into training and validation sets
train_samples, validation_samples = train_test_split(samples_list, test_size=0.2)

# Crate training and validation generators
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

# Model definition based on the LeNet network
model = Sequential()
model.add(Cropping2D(cropping=((60,25), (0,0)), input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))
model.add(Convolution2D(6, 5, 5, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(6, 5, 5, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(120))
model.add(Dropout(0.5))
model.add(Dense(84))
model.add(Dropout(0.5))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.fit_generator(train_generator, samples_per_epoch=len(train_samples),
            validation_data=validation_generator,
            nb_val_samples=len(validation_samples), nb_epoch=5)
model.save('model.h5')