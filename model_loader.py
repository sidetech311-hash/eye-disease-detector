import os
import requests
import streamlit as st

# 🔗 REPLACE THIS URL with your actual direct download link
# (Dropbox, Google Drive direct link, or your own server)
MODEL_URL = "https://your-direct-link-to-model/retinal_final_boss.h5"
MODEL_PATH = "retinal_final_boss.h5"

def download_model_if_missing():
    if not os.path.exists(MODEL_PATH):
        print(f"📡 Model not found. Downloading from {MODEL_URL}...")
        try:
            response = requests.get(MODEL_URL, stream=True)
            response.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("✅ Model downloaded successfully.")
        except Exception as e:
            print(f"❌ Error downloading model: {e}")
            # If download fails, we try to fall back to the v2 model if it exists locally
            if os.path.exists("retinal_disease_model_v2.h5"):
                return "retinal_disease_model_v2.h5"
    return MODEL_PATH
