import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, f1_score, recall_score, precision_score
from tensorflow.keras.preprocessing import image

# Define directories for train and validation sets
train_dir = r"C:\Users\ishni\Downloads\dataset_split\train"       # Path to train data
validation_dir = r"C:\Users\ishni\Downloads\dataset_split\validation"  # Path to validation data

# Check the number of classes in the dataset
train_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
    train_dir,
    target_size=(299, 299),  # InceptionV3 requires 299x299 images
    batch_size=32,
    class_mode='categorical'
)

# Number of classes should match the number of subdirectories in the train directory
num_classes = len(train_generator.class_indices)
print(f"Number of classes detected: {num_classes}")

# Load InceptionV3 model pre-trained on ImageNet, excluding top layers
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

# Freeze layers in the base model
for layer in base_model.layers:
    layer.trainable = False

# Add custom classification layers on top of InceptionV3
x = base_model.output
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
output_layer = Dense(num_classes, activation='softmax')(x)  # Adjusted to match the number of classes

# Define the model
model = Model(inputs=base_model.input, outputs=output_layer)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Set up ImageDataGenerators for data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(299, 299),  # InceptionV3 requires 299x299 images
    batch_size=32,
    class_mode='categorical'
)
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(299, 299),  # InceptionV3 requires 299x299 images
    batch_size=32,
    class_mode='categorical',
    shuffle=False  # Ensure no shuffling for accurate predictions
)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=4,
    validation_data=validation_generator,
    validation_steps=len(validation_generator)
)

# Get true labels and predicted labels for validation set
y_true = validation_generator.classes  # True labels
y_pred_prob = model.predict(validation_generator)  # Predicted probabilities
y_pred = np.argmax(y_pred_prob, axis=1)  # Convert probabilities to class predictions

# Dynamically set class names to avoid any mismatch
class_names = list(validation_generator.class_indices.keys())

# Ensure y_true and y_pred are of the same shape
y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Specify labels explicitly to avoid mismatch
labels = list(range(len(class_names)))

# Calculate and display classification metrics
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names, labels=labels))

# Calculate F1 score, precision, and recall