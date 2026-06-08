import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import os
import json
import numpy as np

# --- CONFIG ---
IMG_SIZE = 224 # Back to 224 because EfficientNet handles memory better
BATCH_SIZE = 8
required_folders = ['ageDegeneration', 'cataract', 'diabetes', 'glaucoma', 'hypertension', 'myopia', 'normal', 'others']

import cv2

# --- IMPROVED PREPROCESSING (CLAHE) ---
def custom_preprocessing(img):
    # Convert to 0-255 range for OpenCV
    img = img.astype(np.uint8)
    # Apply CLAHE to the Green channel (where retinal detail is highest)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return tf.keras.applications.efficientnet.preprocess_input(final_img.astype(np.float32))

# --- DATA GENERATORS (With Oversampling via Class Weights) ---
train_datagen = ImageDataGenerator(
    preprocessing_function=custom_preprocessing,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='constant',
    cval=0,
    validation_split=0.2
)

valid_datagen = ImageDataGenerator(
    preprocessing_function=custom_preprocessing,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    '.',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    classes=required_folders
)

validation_generator = valid_datagen.flow_from_directory(
    '.',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    classes=required_folders
)

# Calculate strong class weights to fight imbalance
class_counts = [266, 293, 1608, 284, 128, 232, 2873, 708] # From our check
total = sum(class_counts)
# Log-based weights are often more stable than linear weights
class_weights = {i: np.log(total / count) for i, count in enumerate(class_counts)}

# --- BUILD MODEL (EfficientNetB0) ---
base_model = tf.keras.applications.EfficientNetB0(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    weights='imagenet',
    include_top=False
)
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(len(required_folders), activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# --- CALLBACKS ---
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6),
    ModelCheckpoint('retinal_efficientnet.keras', monitor='val_accuracy', save_best_only=True)
]

print("🚀 Training EfficientNetB0 (Base)...")
model.fit(
    train_generator,
    epochs=15,
    validation_data=validation_generator,
    class_weight=class_weights,
    callbacks=callbacks
)

# --- UNFREEZE & FINE-TUNE ---
print("🚀 Fine-tuning EfficientNetB0 (Full)...")
base_model.trainable = True
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    train_generator,
    epochs=25,
    validation_data=validation_generator,
    class_weight=class_weights,
    callbacks=callbacks
)

model.save('retinal_disease_model_v2.h5')
print("✅ Done! New model saved as retinal_disease_model_v2.h5")
