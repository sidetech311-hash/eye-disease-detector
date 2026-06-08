# START COPYING HERE
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import os
import json

# --- VERIFY FOLDER STRUCTURE (DO NOT MODIFY) ---
required_folders = [
    'ageDegeneration', 
    'cataract', 
    'diabetes', 
    'glaucoma', 
    'hypertension', 
    'myopia', 
    'normal', 
    'others'
]
missing = [f for f in required_folders if not os.path.exists(f)]
if missing:
    raise FileNotFoundError(
        f"MISSING FOLDERS: {', '.join(missing)}. "
        "Please ensure these 8 folders exist in your current directory."
    )

# --- DATA GENERATORS (AUGMENTATION FOR BETTER GENERALIZATION) ---
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.05,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    '.',  # Current directory
    target_size=(160, 160),  # Reduced from 224 to save memory
    batch_size=8,  # Reduced from 16 to fix OOM
    class_mode='categorical',
    subset='training',
    classes=required_folders  # Only include these 8 folders
)

validation_generator = train_datagen.flow_from_directory(
    '.',
    target_size=(160, 160),  # Reduced from 224
    batch_size=8,  # Reduced from 16
    class_mode='categorical',
    subset='validation',
    classes=required_folders  # Only include these 8 folders
)

# --- SAVE CLASS NAMES FOR API (CRITICAL STEP) ---
class_names = sorted(list(train_generator.class_indices.keys()))
with open('class_names.json', 'w') as f:
    json.dump(class_names, f)
print(f"📝 Saved class names: {class_names}")

# --- BUILD MODEL (MOBILENETV2 + CUSTOM HEAD) ---
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(160, 160, 3),  # Match target_size
    weights='imagenet',
    include_top=False
)
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(
        len(required_folders),
        activation='softmax'
    )
])

# --- COMPILE MODEL ---
model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# --- TRAIN WITH EARLY STOPPING ---
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

print("🚀 Starting training...")
history = model.fit(
    train_generator,
    epochs=50,
    validation_data=validation_generator,
    callbacks=[early_stop]
)

# --- SAVE MODEL & REPORT RESULTS ---
model.save('retinal_disease_model.h5')
val_acc = max(history.history['val_accuracy'])
print(f"✅ MODEL TRAINED SUCCESSFULLY!")
print(f"📊 Validation accuracy: {val_acc:.2%} (Target: ≥75% for retinal tasks)")
print(f"📁 Model saved as: retinal_disease_model.h5")
# END COPYING HERE
