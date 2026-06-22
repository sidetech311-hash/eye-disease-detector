from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import tensorflow as tf
import tf_keras
import cv2
import os
import shutil
import json
import requests

app = FastAPI(title="EyeCare AI API")

# --- 🛡️ CORS FIX (Allows UI to talk to API) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "retinal_final_boss.h5"
MODEL_URL = "https://www.dropbox.com/scl/fi/ruipg8kbuu435c0l73rfp/retinal_disease_model_v2.h5?rlkey=alk3qd9neodv1dehflhej0fcy&st=48evd1oe&dl=1"

def download_model():
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 10000:
        try:
            r = requests.get(MODEL_URL, stream=True)
            with open(MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
        except: pass

download_model()
model = None
classes = []
try:
    with open('class_names.json', 'r') as f: classes = json.load(f)
    model = tf_keras.models.load_model(MODEL_PATH, compile=False)
except: pass

@app.post("/analyze/")
async def analyze_eye(file: UploadFile = File(...)):
    if model is None: raise HTTPException(status_code=503, detail="Brain loading...")
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer: shutil.copyfileobj(file.file, buffer)
    try:
        orig = cv2.imread(temp_path)
        img_res = cv2.resize(orig, (224, 224))
        input_batch = np.expand_dims(tf_keras.applications.efficientnet.preprocess_input(img_res.astype(np.float32)), 0)
        preds = model.predict(input_batch)
        idx = np.argmax(preds[0])
        return {"condition": classes[idx].title(), "confidence": f"{float(preds[0][idx]):.2%}"}
    finally:
        if os.path.exists(temp_path): os.remove(temp_path)

@app.get("/")
def home():
    return {"service": "EyeCare AI API", "status": "Online", "endpoint": "/analyze/"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
