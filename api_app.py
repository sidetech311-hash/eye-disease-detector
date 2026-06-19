from fastapi import FastAPI, File, UploadFile, HTTPException
import numpy as np
import tensorflow as tf
import tf_keras
import cv2
import os
import shutil
import json
import requests

app = FastAPI(title="EyeCare AI API")

MODEL_PATH = "retinal_final_boss.h5"
MODEL_URL = "https://www.dropbox.com/scl/fi/ruipg8kbuu435c0l73rfp/retinal_disease_model_v2.h5?rlkey=alk3qd9neodv1dehflhej0fcy&st=48evd1oe&dl=1"

# --- 🧠 AI BRAIN INITIALIZATION ---
def download_model():
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 10000:
        print("📡 Downloading AI Brain for API...")
        try:
            r = requests.get(MODEL_URL, stream=True)
            with open(MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
            print("✅ Download Complete.")
        except Exception as e:
            print(f"❌ Download Failed: {e}")

download_model()

# Global variables for model and classes
model = None
classes = []

try:
    with open('class_names.json', 'r') as f:
        classes = json.load(f)
    # Use tf_keras for medical model compatibility
    model = tf_keras.models.load_model(MODEL_PATH, compile=False)
except Exception as e:
    print(f"🧠 Brain Load Error: {e}")

def ben_graham_process(img):
    # Standardize size
    img_res = cv2.resize(img, (224, 224))
    return cv2.addWeighted(img_res, 4, cv2.GaussianBlur(img_res, (0,0), 10), -4, 128)

@app.post("/analyze/")
async def analyze_eye(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="AI Brain is offline or loading.")

    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        orig = cv2.imread(temp_path)
        if orig is None:
            raise ValueError("Invalid image file")

        enhanced = ben_graham_process(orig)
        input_batch = np.expand_dims(tf_keras.applications.efficientnet.preprocess_input(enhanced.astype(np.float32)), 0)

        preds = model.predict(input_batch)
        idx = np.argmax(preds[0])
        conf = float(preds[0][idx])
        condition = classes[idx]

        return {
            "condition": condition.title(),
            "confidence": f"{conf:.2%}",
            "status": "Success",
            "message": "Clinical analysis complete"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/")
def home():
    return {
        "service": "EyeCare AI API",
        "status": "Online" if model else "Offline",
        "endpoint": "/analyze/"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
