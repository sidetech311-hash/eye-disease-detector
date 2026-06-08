import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import numpy as np
import json
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

import cv2

def custom_preprocess(img):
    img = img.astype(np.uint8)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return tf.keras.applications.efficientnet.preprocess_input(final_img.astype(np.float32))

# Load model & data
MODEL_NAME = 'retinal_final_boss.h5'
if not os.path.exists(MODEL_NAME):
    MODEL_NAME = 'retinal_disease_model_v2.h5'

with open('class_names.json', 'r') as f:
    class_names = json.load(f)

# Load without compiling
model = load_model(MODEL_NAME, compile=False)

datagen = ImageDataGenerator(preprocessing_function=custom_preprocess)
generator = datagen.flow_from_directory(
    '.',
    target_size=(192, 192),
    batch_size=4,
    class_mode='categorical',
    shuffle=False,
    classes=class_names
)

# Get predictions & true labels
preds = model.predict(generator)
y_pred = np.argmax(preds, axis=1)
y_true = generator.classes

# Generate reports
print("=== CLASSIFICATION REPORT ===")
print(classification_report(y_true, y_pred, target_names=class_names))

# Plot confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Retinal Disease Classification Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
print("\n✅ Confusion matrix saved as 'confusion_matrix.png'")
