import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import matplotlib.pyplot as plt

# PATH = os.path.dirname(os.path.abspath('cats_and_dogs/'))

cwd = os.getcwd()

PATH = 'cats_and_dogs'

train_dir = os.path.join(cwd, 'apps', 'AI', 'cats_and_dogs', 'train')
validation_dir = os.path.join(cwd, 'apps', 'AI', 'cats_and_dogs', 'validation')
test_dir = os.path.join(cwd, 'apps', 'AI', 'cats_and_dogs', 'test')

print(train_dir)

# Get number of files in each directory. The train and validation directories
# each have the subdirecories "dogs" and "cats".
total_train = sum([len(files) for r, d, files in os.walk(train_dir)])
total_val = sum([len(files) for r, d, files in os.walk(validation_dir)])
total_test = sum([len(files) for r, d, files in os.walk(test_dir)])

# Variables for pre-processing and training.
batch_size = 128
epochs = 15
IMG_HEIGHT = 150
IMG_WIDTH = 150

train_image_generator = ImageDataGenerator(rescale=1./255, rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
validation_image_generator = ImageDataGenerator(rescale=1./255)
test_image_generator = ImageDataGenerator(rescale=1./255)

train_data_gen = train_image_generator.flow_from_directory(train_dir, batch_size=32, target_size=(256, 256), class_mode='binary')
val_data_gen = validation_image_generator.flow_from_directory(validation_dir, batch_size=32, target_size=(256, 256), class_mode='binary')
test_data_gen = test_image_generator.flow_from_directory(test_dir, batch_size=32, target_size=(256, 256), class_mode='binary')


