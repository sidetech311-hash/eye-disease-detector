from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import json
import os

app = Flask(__name__)

# Load model & classes
MODEL_PATH = 'retinal_disease_fine_tuned.h5' if os.path.exists('retinal_disease_fine_tuned.h5') else 'retinal_disease_model.h5'
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"✅ Loaded model from {MODEL_PATH}")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    model = None

with open('class_names.json', 'r') as f:
    class_names = json.load(f)

@app.route('/')
def index():
    return '''
    <h1>Retinal Disease Screening Tool</h1>
    <p>Upload a fundus photo to screen for retinal conditions</p>
    <form action="/predict" method="post" enctype="multipart/form-data">
        <input type="file" name="file" accept="image/*" required>
        <button type="submit">Analyze</button>
    </form>
    <div id="result">{{ prediction_text }}</div>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    # Process image
    img_path = f"temp_{file.filename}"
    file.save(img_path)
    img = image.load_img(img_path, target_size=(160, 160))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    preds = model.predict(img_array)[0]
    pred_class = class_names[np.argmax(preds)]
    confidence = float(np.max(preds))
    
    # Cleanup
    os.remove(img_path)
    
    return f"""
    <h1>Results</h1>
    <p><strong>Condition:</strong> {pred_class.replace('_', ' ').title()}</p>
    <p><strong>Confidence:</strong> {confidence:.1%}</p>
    <p><em>Note: This is a screening tool — consult a specialist for diagnosis</em></p>
    <a href="/">Try another image</a>
    """

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
