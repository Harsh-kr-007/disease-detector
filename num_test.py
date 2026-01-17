import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, f1_score, recall_score, precision_score, confusion_matrix
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import seaborn as sns


MODEL_SAVE_FILENAME = 'covid_classifier_model.h5'
CLASS_INDICES_FILENAME = 'class_indices.json'
TARGET_SIZE = (224, 224)

train_dir = r"C:\Users\ishni\Downloads\dataset_split\train"
validation_dir = r"C:\Users\ishni\Downloads\dataset_split\validation"


base_model = VGG19(weights='imagenet', include_top=False, input_shape=(TARGET_SIZE[0], TARGET_SIZE[1], 3))


for layer in base_model.layers[:-4]:
    layer.trainable = False


train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=TARGET_SIZE,
    batch_size=32,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=TARGET_SIZE,
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# --- SAVE CLASS INDICES ---
class_indices = train_generator.class_indices
with open(CLASS_INDICES_FILENAME, 'w') as f:
    json.dump(class_indices, f)
print(f"✅ Saved class indices mapping to '{CLASS_INDICES_FILENAME}': {class_indices}")

num_classes = len(class_indices)

# --- CUSTOM TOP LAYERS ---
x = base_model.output
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
output_layer = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output_layer)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# --- CLASS WEIGHTS (HANDLE IMBALANCE) ---
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights = dict(enumerate(class_weights))
print(f"🧮 Computed class weights: {class_weights}")

# --- CALLBACKS ---
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True)
]

# --- TRAIN MODEL ---
print("\n🚀 Starting Model Training...")
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=25,
    callbacks=callbacks,
    class_weight=class_weights
)

# --- SAVE TRAINED MODEL ---
model.save(MODEL_SAVE_FILENAME)
print(f"\n✅ Model saved successfully as '{MODEL_SAVE_FILENAME}'")

# --- EVALUATION ---
print("\n📊 Evaluating Model...")
y_true = validation_generator.classes
y_pred_prob = model.predict(validation_generator)
y_pred = np.argmax(y_pred_prob, axis=1)
class_names = [k for k, v in sorted(class_indices.items(), key=lambda item: item[1])]

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))
print(f"Weighted F1: {f1_score(y_true, y_pred, average='weighted'):.2f}")
print(f"Weighted Recall: {recall_score(y_true, y_pred, average='weighted'):.2f}")
print(f"Weighted Precision: {precision_score(y_true, y_pred, average='weighted'):.2f}")

# --- CONFUSION MATRIX ---
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# --- SINGLE IMAGE TEST ---
print("\n🖼️ Testing single image...")
img_path = r"C:\Users\ishni\Downloads\dataset-cover (1).png"
img = image.load_img(img_path, target_size=TARGET_SIZE)
img = img.convert('RGB')
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

single_prediction = model.predict(img_array)
predicted_class_index = np.argmax(single_prediction)
predicted_class_label = class_names[predicted_class_index]
confidence = np.max(single_prediction)

print(f"Predicted Class: {predicted_class_label} (Confidence: {confidence:.2f})")
