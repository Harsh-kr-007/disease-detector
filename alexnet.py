import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, f1_score, recall_score, precision_score
from tensorflow.keras.preprocessing import image


train_dir = r"C:\Users\ishni\Downloads\dataset_split\train"       
validation_dir = r"C:\Users\ishni\Downloads\dataset_split\validation"  


def alexnet(input_shape=(224, 224, 3), num_classes=5):
    model = models.Sequential()
    
    
    model.add(layers.Conv2D(96, (11, 11), strides=(4, 4), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    
    
    model.add(layers.Conv2D(256, (5, 5), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    
    
    model.add(layers.Conv2D(384, (3, 3), padding='same', activation='relu'))
    
    
    model.add(layers.Conv2D(384, (3, 3), padding='same', activation='relu'))
    
    
    model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    
    
    model.add(layers.Flatten())
    
   
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))
    
    
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    return model


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
    shuffle=False  
)


num_classes = len(train_generator.class_indices)


model = alexnet(input_shape=(224, 224, 3), num_classes=num_classes)


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=4,
    validation_data=validation_generator,
    validation_steps=len(validation_generator)
)


y_true = validation_generator.classes  
y_pred_prob = model.predict(validation_generator)  
y_pred = np.argmax(y_pred_prob, axis=1)  


class_names = list(validation_generator.class_indices.keys())


y_true = np.array(y_true)
y_pred = np.array(y_pred)


labels = list(range(len(class_names)))


print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names, labels=labels))


f1 = f1_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
precision = precision_score(y_true, y_pred, average='weighted')


print(f"Weighted F1 Score: {f1:.2f}")
print(f"Weighted Recall: {recall:.2f}")
print(f"Weighted Precision: {precision:.2f}")


img_path = r"C:\Users\ishni\Downloads\dataset-cover (1).png"  


img = image.load_img(img_path, target_size=(224, 224))
img = img.convert('RGB')  
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0  


single_prediction = model.predict(img_array)
predicted_class_index = np.argmax(single_prediction)
predicted_class_label = class_names[predicted_class_index]

print(f"Predicted Class for the test image: {predicted_class_label}")
