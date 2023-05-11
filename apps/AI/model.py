import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np


# Define the working directory into the `cwd` variable
cwd = os.getcwd()

# Path to the directory containing the images used
PATH = os.path.join(cwd, 'apps', 'AI', 'cats_and_dogs')

# Define the paths to the train, validation and test directories within cats_and_dogs
train_dir = os.path.join(cwd, 'apps', 'AI', 'cats_and_dogs', 'train')
validation_dir = os.path.join(cwd, 'apps', 'AI', 'cats_and_dogs', 'validation')
test_dir = os.path.join(cwd, 'apps', 'AI', 'cats_and_dogs', 'test')

# Count the total number of files 
total_train = sum([len(files) for r, d, files in os.walk(train_dir)])
total_val = sum([len(files) for r, d, files in os.walk(validation_dir)])
total_test = sum([len(files) for r, d, files in os.walk(test_dir)])


# Set the batch size for training
batch_size = 128

#Set the number of epochs the AI will be running
epochs = 30

# Define the image proportion
IMG_HEIGHT = 150
IMG_WIDTH = 150

# Create image data generators and rescale the images
train_image_generator = ImageDataGenerator(rescale=1./255)
validation_image_generator = ImageDataGenerator(rescale=1./255)
test_image_generator = ImageDataGenerator(rescale=1./255)

# Generate training data from the images in the train directory
train_data_gen = train_image_generator.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=batch_size,
    class_mode="binary",
)

# Generate validation data from the images in the validation directory
val_data_gen = validation_image_generator.flow_from_directory(
    validation_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=batch_size,
    class_mode="binary",
    seed=42  # Set a seed for reproducibility
)

# Generate test data from the images in the PATH directory
test_data_gen = test_image_generator.flow_from_directory(
    PATH,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=1,
    classes=['test'],
    shuffle=False, # Disable shuffling to maintain order
)

# Define an image data generator for data augmentation during training
train_image_generator = ImageDataGenerator(
    rescale=1./255,  # Rescale pixel values to [0, 1]
    rotation_range=40,  # Randomly rotate images by 40 degrees
    width_shift_range=0.2,  # Randomly shift images horizontally by 20% of the width
    height_shift_range=0.2,  # Randomly shift images vertically by 20% of the height
    shear_range=0.2,  # Apply random shearing transformations
    zoom_range=0.2,  # Apply random zooming transformations
    horizontal_flip=True,  # Randomly flip images horizontally
    fill_mode='nearest'  # Use the nearest pixel for filling newly created pixels
)

# Generate augmented training data from the images in the train directory
train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
    directory=train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode='binary')

# Select and store a sample of augmented images to precent overfitting
augmented_images = [train_data_gen[0][0][0] for i in range(5)]


# Create a sequential model
model = Sequential([
    Conv2D(16, (3,3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),  # Convolutional layer with 16 filters, 3x3 kernel, ReLU activation, and input shape
    MaxPooling2D(2, 2),  # Max pooling layer with 2x2 pool size
    Conv2D(32, (3,3), activation='relu'),  # Convolutional layer with 32 filters and 3x3 kernel
    MaxPooling2D(2,2),  # Max pooling layer with 2x2 pool size
    Conv2D(64, (3,3), activation='relu'),  # Convolutional layer with 64 filters and 3x3 kernel
    MaxPooling2D(2,2),  # Max pooling layer with 2x2 pool size
    Flatten(),  # Flatten the output for dense layers
    Dense(512, activation='relu'),  # Fully connected dense layer with 512 units and ReLU activation
    Dense(1, activation='sigmoid')  # Output layer with 1 unit and sigmoid activation for binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_data_gen,
    steps_per_epoch=train_data_gen.n // train_data_gen.batch_size,  # Set the number of steps per epoch based on the training data generator
    epochs=epochs,  # Set the number of epochs for training
    validation_data=val_data_gen,
    validation_steps=val_data_gen.n // val_data_gen.batch_size  # Set the number of steps for validation based on the validation data generator
)

# Save the trained model
model.save('cat_dog_classifier.h5')
