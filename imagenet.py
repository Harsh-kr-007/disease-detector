import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, f1_score, recall_score, precision_score
from tensorflow.keras.preprocessing import image

# Define directories for train and validation sets
train_dir = r"C:\Users\ishni\Downloads\dataset_split\train"       # Path to train data
validation_dir = r"C:\Users\ishni\Downloads\dataset_split\validation"  # Path to validation data

# Load VGG19 model pre-trained on ImageNet, excluding top layers
base_model = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze layers in the base model to retain pre-trained features
for layer in base_model.layers:
    layer.trainable = False

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
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False  # Ensure no shuffling for accurate predictions
)

# Automatically set the number of classes based on training data
num_classes = len(train_generator.class_indices)

# Add custom classification layers on top of the base VGG19 model
x = base_model.output
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
output_layer = Dense(num_classes, activation='softmax')(x)  # Output layer with 'num_classes' units

# Define the model
model = Model(inputs=base_model.input, outputs=output_layer)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

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

# Calculate F1 score, precision, and recall for each class
f1 = f1_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
precision = precision_score(y_true, y_pred, average='weighted')

# Print out F1 score, recall, and precision
print(f"Weighted F1 Score: {f1:.2f}")
print(f"Weighted Recall: {recall:.2f}")
print(f"Weighted Precision: {precision:.2f}")

# Example of predicting a single image
img_path = r"C:\Users\ishni\Downloads\dataset-cover (1).png"  # Path to the image for prediction

# Ensure the image is converted to 3 channels (RGB)
img = image.load_img(img_path, target_size=(224, 224))
img = img.convert('RGB')  # Force the image to be in RGB mode if it's not
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0  # Normalize the image

# Make a prediction
single_prediction = model.predict(img_array)
predicted_class_index = np.argmax(single_prediction)
predicted_class_label = class_names[predicted_class_index]

# Output the prediction for the single image
print(f"Predicted Class for the test image: {predicted_class_label}")