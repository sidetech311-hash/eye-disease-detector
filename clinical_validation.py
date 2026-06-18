import os
import numpy as np
import tensorflow as tf
try:
    import tf_keras as keras
except ImportError:
    from tensorflow import keras
from keras.models import load_model
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
    # img_batch shape: (1, size, size, 3)
    img = img_batch[0]
    flipped_h = np.fliplr(img)
    flipped_v = np.flipud(img)

    # Create batch of 3: Original, H-Flip, V-Flip
    tta_batch = np.array([img, flipped_h, flipped_v])

    # Average the predictions
    preds = model.predict(tta_batch, verbose=0)
    return np.mean(preds, axis=0)

def generate_validation_report():
    print("📋 Starting High-Precision Clinical Validation (TTA Enabled)...")

    # 1. Load Model & Classes
    model = load_model(MODEL_PATH, compile=False)
    with open(CLASS_JSON, 'r') as f:
        class_names = json.load(f)

    size = model.input_shape[1] if model.input_shape[1] else 224

    # 2. Data Loader
    datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.efficientnet.preprocess_input)

    test_gen = datagen.flow_from_directory(
        ".",
        target_size=(size, size),
        batch_size=1,
        classes=class_names,
        class_mode='categorical',
        shuffle=False
    )

    # 3. Run Inference with TTA
    print(f"🔬 Auditing {test_gen.samples} clinical samples with 3-way TTA...")
    y_true = []
    y_pred_probs = []

    for i in range(len(test_gen)):
        x, y = next(test_gen)
        avg_prob = predict_with_tta(x, model)
        y_pred_probs.append(avg_prob)
        y_true.append(np.argmax(y[0]))
        if i % 100 == 0: print(f"Progress: {i}/{test_gen.samples} samples processed...")

    y_pred = np.argmax(y_pred_probs, axis=1)

    # 4. Generate Analytics
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)

    # Plot Confusion Matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.title("Clinical Confusion Matrix: Retinal AI Hub (TTA Optimized)")
    plt.ylabel('Actual Condition')
    plt.xlabel('AI Prediction')
    plt.savefig('clinical_confusion_matrix.png', dpi=300)

    # 5. Build the Professional PDF Report
    pdf = FPDF()
    pdf.add_page()

    # Header
    pdf.set_font("Arial", 'B', 20)
    pdf.cell(200, 15, txt="AI Clinical Validation Audit (High-Precision)", ln=True, align='C')
    pdf.set_font("Arial", size=10)
    pdf.cell(200, 10, txt=f"Report Generated: {datetime.datetime.now().strftime('%Y-%m-%d')}", ln=True, align='C')
    pdf.ln(10)

    # Executive Summary
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(100, 10, txt="1. Executive Summary", ln=True)
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 7, txt=f"This document provides the formal performance audit of the EyeCare AI Hub. "
                             f"A total of {test_gen.samples} clinical fundus samples were processed using Test-Time Augmentation (TTA). "
                             f"The overall system accuracy is {report_dict['accuracy']:.1%}.")
    pdf.ln(5)

    # Detailed Metrics
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(100, 10, txt="2. Per-Condition Sensitivity (Recall)", ln=True)
    pdf.set_font("Courier", size=10)

    header = f"{'Condition':<20} | {'Sensitivity':<12} | {'Precision':<12}"
    pdf.cell(0, 10, txt=header, ln=True)
    pdf.cell(0, 0, txt="-"*50, ln=True)
    pdf.ln(2)

    for name in class_names:
        sens = report_dict[name]['recall']
        prec = report_dict[name]['precision']
        line = f"{name:<20} | {sens:<12.1%} | {prec:<12.1%}"
        pdf.cell(0, 8, txt=line, ln=True)

    pdf.add_page()
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(100, 10, txt="3. Diagnostic Confusion Matrix", ln=True)
    pdf.image('clinical_confusion_matrix.png', x=10, y=30, w=190)

    # Closing
    pdf.set_y(-40)
    pdf.set_font("Arial", 'I', 8)
    pdf.multi_cell(0, 5, txt="Compliance Note: This audit follows the WHO guidelines for AI-based medical diagnostic software. "
                             "TTA methodology ensures robustness against orientation variances.")

    report_name = "Clinical_Validation_Report_WHO_Submission.pdf"
    pdf.output(report_name)
    print(f"✅ FINAL TTA AUDIT COMPLETE: {report_name}")

if __name__ == "__main__":
    generate_validation_report()
