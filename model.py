import csv
import cv2
import numpy as np
import sklearn

default_batch_size = 30

### Generator and Image Processing
def generator(data, batch_size=default_batch_size):
  path = './data/IMG/'
  while 1: # Loop forever so the generator never terminates
    for i in range(0, len(data), batch_size):
      batch = data[i:i+batch_size]
      images = []
      measurements = []
      for row in batch:
        center_image = cv2.imread(path + row[0].split('/')[-1])
        center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
        images.append(center_image)

        left_image = cv2.imread(path + row[1].split('/')[-1])
        left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
        images.append(left_image)

        right_image = cv2.imread(path + row[2].split('/')[-1])
        right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)
        images.append(right_image)

        correction = 0.05
        steering_angle = float(row[3])
        measurements.append(steering_angle)
        measurements.append(steering_angle + correction)
        measurements.append(steering_angle - correction)

      augmented_images, augmented_measurements = augment(images, measurements)
      X = np.array(augmented_images)
      y = np.array(augmented_measurements)
      yield sklearn.utils.shuffle(X, y)
# Flip images for augmented data
def augment(images, measurements):
  augmented_images, augmented_measurements = [], []
  for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image, 1))
    augmented_measurements.append(measurement * -1.0)
  return augmented_images, augmented_measurements


### Model Architecture
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Lambda, Cropping2D, Convolution2D

model = Sequential()
# Normalization
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
# Cropping
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
# Convolutional Layers
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
# Fully Connected Layers
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.3))
model.add(Dense(50))
model.add(Dropout(0.3))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')


### Model Training
from keras.models import Model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data = []
with open('./data/driving_log.csv') as csvfile:
  reader = csv.reader(csvfile)
  for line in reader:
    data.append(line)
data = data[1:]  # remove csv title line

train_samples, validation_samples = train_test_split(data, test_size=0.2)
train_generator = generator(train_samples)
validation_generator = generator(validation_samples)

model.fit_generator(train_generator, 
    samples_per_epoch = len(train_samples), 
    validation_data = validation_generator,
    nb_val_samples = len(validation_samples), 
    nb_epoch=5, 
    verbose=1)

model.save('model.h5')