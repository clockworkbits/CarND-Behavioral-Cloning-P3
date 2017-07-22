import csv
import cv2
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

# Load the data from the csv file
print("Loading the data from the csv file")
lines = []
with open("data/driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

correction_factor = 0.2

# Load the images (from all three cameras)
print("Loading the images")        
images = []
measurments = []
for line in lines:
    measurement = float(line[3])
    # Drop every other sample where the steering angle is close to 0
    if (measurement < 0.025 and measurement > -0.025 and random.choice([True, False])):
        continue

    for i in range(3):
        source_path = line[i] # 0 index is for the center camera
        filename = source_path.split('/')[-1]
        current_path = 'data/IMG/' + filename
        image = cv2.imread(current_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # convert to RGB
        images.append(image)

        # Add the measurement for the given image
        if i == 0: # Center
            measurments.append(measurement)
        elif i == 1: # Left
            measurments.append(measurement + correction_factor)
        else: # i == 2 - Right
            measurments.append(measurement - correction_factor)

# Augment data by flipping the image horizontally
print("Augmenting the data")
aug_images = []
aug_measurments = []

for image, measurment in zip(images, measurments):
    aug_images.append(image)
    aug_measurments.append(measurment)
    aug_images.append(cv2.flip(image, flipCode=1))
    aug_measurments.append(-measurment)

# Create the training data
X_train = np.array(aug_images)
y_train = np.array(aug_measurments)

# Display the number of training examples
print("Number of training examples =", len(X_train))

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
history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)
model.save('model.h5')