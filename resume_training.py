import os
# Disable oneDNN to prevent "could not create a primitive" crashes
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.models import load_model
import numpy as np
import cv2

# --- CONFIG (Same as v3) ---
IMG_SIZE = 192
BATCH_SIZE = 4
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

# --- FOCAL LOSS ---
def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -tf.reduce_sum(alpha * tf.pow(1. - pt_1, gamma) * tf.math.log(pt_1)) \
               -tf.reduce_sum((1 - alpha) * tf.pow(pt_0, gamma) * tf.math.log(1. - pt_0))
    return focal_loss_fixed

# --- LOAD BEST MODEL ---
print("📂 Loading previous best checkpoint...")
model = load_model('retinal_final_boss.h5', custom_objects={'focal_loss_fixed': focal_loss(gamma=2.0)})

# --- CALLBACKS ---
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-8),
    ModelCheckpoint('retinal_final_boss.h5', monitor='val_accuracy', save_best_only=True)
]

print("🚀 Resuming training...")
model.fit(
    train_gen,
    initial_epoch=27, # Start from where it crashed
    epochs=40,
    validation_data=valid_gen,
    callbacks=callbacks
)

print("✅ Training finished! Best model updated in retinal_final_boss.h5")
