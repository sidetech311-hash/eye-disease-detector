import os
import numpy as np
import tensorflow as tf
# Force Keras 2.0 Compatibility
import tf_keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import json
import datetime

# --- CONFIG ---
MODEL_PATH = "retinal_final_boss.h5"
CLASS_JSON = "class_names.json"

def predict_with_tta(img_batch, model):
    """Applies TTA (Horizontal & Vertical Flips) and averages results."""
    img = img_batch[0]
    flipped_h = np.fliplr(img)
    flipped_v = np.flipud(img)
    tta_batch = np.array([img, flipped_h, flipped_v])
    preds = model.predict(tta_batch, verbose=0)
    return np.mean(preds, axis=0)

def generate_validation_report():
    print("📋 Starting High-Precision Clinical Validation (Keras 2 Compatibility Mode)...")

    # 1. Load Model & Classes using tf_keras
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: {MODEL_PATH} not found!")
        return

    model = tf_keras.models.load_model(MODEL_PATH, compile=False)

    with open(CLASS_JSON, 'r') as f:
        class_names = json.load(f)

    # Detect input size from model
    size = model.input_shape[1] if model.input_shape[1] else 224

    # 2. Data Loader
    # Note: Ensure you have your test images organized in folders by class name
    datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.efficientnet.preprocess_input)

    # Use the current directory or specify your dataset path here
    data_path = "."

    try:
        test_gen = datagen.flow_from_directory(
            data_path,
            target_size=(size, size),
            batch_size=1,
            classes=class_names,
            class_mode='categorical',
            shuffle=False
        )
    except Exception as e:
        print(f"❌ Data Error: {e}")
        return

    # 3. Run Inference
    print(f"🔬 Auditing {test_gen.samples} samples...")
    y_true = []
    y_pred_probs = []

    for i in range(len(test_gen)):
        x, y = next(test_gen)
        avg_prob = predict_with_tta(x, model)
        y_pred_probs.append(avg_prob)
        y_true.append(np.argmax(y[0]))
        if i % 100 == 0: print(f"Progress: {i}/{test_gen.samples} processed...")

    y_pred = np.argmax(y_pred_probs, axis=1)

    # 4. Generate Analytics
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.title("Clinical Confusion Matrix (TTA Optimized)")
    plt.ylabel('Actual Condition')
    plt.xlabel('AI Prediction')
    plt.savefig('clinical_confusion_matrix.png', dpi=300)

    # 5. Build PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 20)
    pdf.cell(200, 15, txt="AI Clinical Validation Audit", ln=True, align='C')
    pdf.set_font("Arial", size=10)
    pdf.cell(200, 10, txt=f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d')}", ln=True, align='C')
    pdf.ln(10)

    pdf.set_font("Arial", 'B', 14)
    pdf.cell(100, 10, txt="1. Executive Summary", ln=True)
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 7, txt=f"Overall System Accuracy: {report_dict['accuracy']:.1%}")

    pdf.ln(10)
    pdf.image('clinical_confusion_matrix.png', x=10, y=70, w=180)

    report_name = "Clinical_Validation_Report_WHO_Submission.pdf"
    pdf.output(report_name)
    print(f"✅ AUDIT COMPLETE: {report_name}")

if __name__ == "__main__":
    generate_validation_report()
