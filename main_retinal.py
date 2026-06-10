# START COPY
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import json
import shutil
import os

import cv2
from gradcam_utils import make_gradcam_heatmap, save_and_display_gradcam
from model_loader import download_model_if_missing

# --- AUTO-DOWNLOAD & LOAD MODEL ---
MODEL_PATH = download_model_if_missing()

try:
    # Load without compiling to avoid custom loss requirement
    model = load_model(MODEL_PATH, compile=False)
    with open('class_names.json', 'r') as f:
        class_names = json.load(f)
    print(f"✅ Model loaded from {MODEL_PATH}. Classes: {class_names}")
except Exception as e:
    print(f"❌ Model load failed: {e}")
    model = None
    class_names = []

app = FastAPI(title="Retinal Disease Detector")

# --- PREPROCESSING MATCH (CRITICAL) ---
def preprocess_for_model(img_path):
    # Load image
    img = image.load_img(img_path, target_size=(192, 192))
    img_array = image.img_to_array(img).astype(np.uint8)

    # Apply CLAHE (must match training exactly)
    lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

    # EfficientNet scaling
    final_img = tf.keras.applications.efficientnet.preprocess_input(final_img.astype(np.float32))
    return np.expand_dims(final_img, axis=0)

def predict_disease(img_path):
    if model is None: 
        return "model_error", 0.0, None

    img_tensor = preprocess_for_model(img_path)
    preds = model.predict(img_tensor)
    class_idx = np.argmax(preds[0])
    confidence = float(np.max(preds[0]))

    # Generate Grad-CAM heatmap
    try:
        # For EfficientNetB0, the last conv layer is 'top_activation'
        heatmap = make_gradcam_heatmap(img_tensor, model, "top_activation")
        cam_filename = f"cam_{os.path.basename(img_path)}"
        cam_full_path = os.path.join(static_dir, cam_filename)
        save_and_display_gradcam(img_path, heatmap, cam_path=cam_full_path)
        gradcam_url = f"/static/{cam_filename}"
    except Exception as e:
        print(f"Grad-CAM error: {e}")
        gradcam_url = None

    # "I don't know" logic for low confidence
    if confidence < 0.35:
        return "Uncertain / Requires Review", confidence, gradcam_url

    return class_names[class_idx], confidence, gradcam_url

@app.post("/detect/")
async def detect(file: UploadFile = File(...)):
    # Validate file type
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Only PNG/JPG allowed")

    # Save uploaded file temporarily
    temp_file = f"temp_{file.filename}"
    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Run prediction
        disease, confidence, gradcam_url = predict_disease(temp_file)

        # Prepare response
        return JSONResponse({
            "disease": disease,
            "confidence": round(confidence, 3),
            "gradcam_url": gradcam_url,
            "status": "success" if disease != "model_error" else "error"
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Clean up temp file
        if os.path.exists(temp_file):
            os.remove(temp_file)

# main_retinal.py  (add near the top, after imports)
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import pathlib

# -----------------------------------------------------------------
# Serve a simple HTML UI from the /ui/templates folder
# -----------------------------------------------------------------
BASE_DIR = pathlib.Path(__file__).parent
templates_dir = BASE_DIR / "ui" / "template"
static_dir    = BASE_DIR / "ui" / "static"

# Create static directory if it doesn't exist to avoid errors
if not static_dir.exists():
    os.makedirs(static_dir, exist_ok=True)

app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/", response_class=HTMLResponse)
async def root():
    html_file = templates_dir / "index.html"
    if html_file.exists():
        return HTMLResponse(html_file.read_text(encoding="utf-8"))
    return {
        "message": "Retinal Disease Detector API is running",
        "model_loaded": model is not None,
        "classes": class_names if model else [],
        "note": "Send POST request to /detect/ with fundus photo. UI template not found."
    }

# --- RUN SERVER (if executed directly) ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
# END COPY
