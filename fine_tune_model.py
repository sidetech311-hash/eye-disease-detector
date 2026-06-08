# fine_tune_model.py
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import os
import json

# --- LOAD EXISTING MODEL ---
model = load_model('retinal_disease_model.h5')

# --- LOAD CLASS NAMES ---
with open('class_names.json', 'r') as f:
    class_names = json.load(f)

# --- UNFREEZE ALL LAYERS ---
base_model = model.layers[0]
base_model.trainable = True

# --- RECOMPILE WITH CORRECT PREPROCESSING & STABLE LR ---
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# --- DATA GENERATORS (Corrected Preprocessing & Size) ---
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

IMG_SIZE = 192
BATCH_SIZE = 4 # Small batch to avoid OOM at 192x192

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input, # Use official MobileNetV2 scaling
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='constant',
    cval=0,
    validation_split=0.2
)

valid_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    '.',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    classes=class_names,
    shuffle=True
)

validation_generator = valid_datagen.flow_from_directory(
    '.',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    classes=class_names,
    shuffle=False
)

# --- CALLBACKS ---
early_stop = EarlyStopping(monitor='val_accuracy', patience=7, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7)
checkpoint = ModelCheckpoint('retinal_disease_fine_tuned.keras', monitor='val_accuracy', save_best_only=True, verbose=1)

# --- MILDER CLASS WEIGHTS ---
# Extreme weights can cause the model to diverge. We use the square root to "soften" them.
class_counts = []
for name in class_names:
    count = len([f for f in os.listdir(name) if f.endswith(('.jpg', '.jpeg', '.png'))])
    class_counts.append(max(count, 1))
total = sum(class_counts)
import numpy as np
# Square root smoothing helps keep the weights from being too extreme
class_weights = {i: np.sqrt(total/(len(class_names)*count)) for i, count in enumerate(class_counts)}

print("🚀 Starting STABLE fine-tuning...")
history = model.fit(
    train_generator,
    epochs=40,
    validation_data=validation_generator,
    callbacks=[early_stop, reduce_lr, checkpoint],
    class_weight=class_weights
)

# --- FINAL REPORT ---
val_acc = max(history.history['val_accuracy'])
print(f"✅ FINE-TUNING COMPLETE!")
print(f"📊 Best Validation accuracy: {val_acc:.2%}")
