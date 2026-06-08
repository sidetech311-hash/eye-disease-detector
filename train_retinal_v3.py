import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import os
import json
import numpy as np
import cv2

# --- CONFIG ---
IMG_SIZE = 192 # Reduced from 224 to save memory
BATCH_SIZE = 4 # Reduced from 8 to avoid OOM
CLASSES = ['ageDegeneration', 'cataract', 'diabetes', 'glaucoma', 'hypertension', 'myopia', 'normal', 'others']

# --- CLAHE PREPROCESSING ---
def custom_preprocessing(img):
    img = img.astype(np.uint8)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return tf.keras.applications.efficientnet.preprocess_input(final_img.astype(np.float32))

# --- DATA GENERATORS ---
train_datagen = ImageDataGenerator(
    preprocessing_function=custom_preprocessing,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    validation_split=0.2
)

valid_datagen = ImageDataGenerator(
    preprocessing_function=custom_preprocessing,
    validation_split=0.2
)

train_gen = train_datagen.flow_from_directory('.', target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, classes=CLASSES, subset='training')
valid_gen = valid_datagen.flow_from_directory('.', target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, classes=CLASSES, subset='validation', shuffle=False)

# --- FOCAL LOSS (The Secret Weapon) ---
def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -tf.reduce_sum(alpha * tf.pow(1. - pt_1, gamma) * tf.math.log(pt_1)) \
               -tf.reduce_sum((1 - alpha) * tf.pow(pt_0, gamma) * tf.math.log(1. - pt_0))
    return focal_loss_fixed

# --- MODEL BUILD ---
base_model = tf.keras.applications.EfficientNetB0(input_shape=(IMG_SIZE, IMG_SIZE, 3), weights='imagenet', include_top=False)
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(len(CLASSES), activation='softmax')
])

# --- STAGE 1: WARMUP ---
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
print("🚀 Stage 1: Warming up top layers...")
model.fit(train_gen, epochs=5, validation_data=valid_gen)

# --- STAGE 2: FULL FINE-TUNE WITH FOCAL LOSS ---
base_model.trainable = True
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss=focal_loss(gamma=2.0), metrics=['accuracy'])

callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7),
    ModelCheckpoint('retinal_final_boss.h5', monitor='val_accuracy', save_best_only=True)
]

print("🚀 Stage 2: Deep training with Focal Loss...")
model.fit(train_gen, epochs=35, validation_data=valid_gen, callbacks=callbacks)

print("✅ Final model saved as retinal_final_boss.h5")
