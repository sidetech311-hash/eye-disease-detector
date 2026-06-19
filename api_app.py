from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
import numpy as np
import tensorflow as tf
try:
    import tf_keras as keras
except ImportError:
    from tensorflow import keras
from keras.models import load_model
import cv2
import os
import shutil
import json

app = FastAPI(title="EyeCare AI API")

MODEL_PATH = "retinal_final_boss.h5"
with open('class_names.json', 'r') as f:
    classes = json.load(f)

# Load model once
model = load_model(MODEL_PATH, compile=False)

def ben_graham_process(img):
    img_res = cv2.resize(img, (224, 224))
    return cv2.addWeighted(img_res, 4, cv2.GaussianBlur(img_res, (0,0), 10), -4, 128)

@app.post("/analyze/")
async def analyze_eye(file: UploadFile = File(...)):
    # 1. Save temp file
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # 2. Process
        orig = cv2.imread(temp_path)
        enhanced = ben_graham_process(orig)
        input_batch = np.expand_dims(tf.keras.applications.efficientnet.preprocess_input(enhanced.astype(np.float32)), 0)

        # 3. Predict
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
    return {"message": "EyeCare AI API is Live. Use /analyze/ for Postman tests."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
