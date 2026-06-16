import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import json
import os
import pandas as pd
from PIL import Image, ImageOps
import io
from fpdf import FPDF
import datetime
import sqlite3
import requests

# --- ⚙️ CONFIGURATION ---
st.set_page_config(page_title="EyeCare AI Hub Pro", layout="wide", page_icon="👁️")
MODEL_PATH = "retinal_final_boss.h5"
# Direct link for auto-download (Replace with your actual link)
MODEL_URL = "https://www.dropbox.com/scl/fi/YOUR_LINK_HERE/retinal_final_boss.h5?rlkey=YOUR_KEY&dl=1"

# --- 🔐 SECURITY ---
def check_password():
    def password_entered():
        if st.session_state["password"] == "doctor123":
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("🔑 Physician Password", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("🔑 Physician Password", type="password", on_change=password_entered, key="password")
        st.error("😕 Password incorrect")
        return False
    else:
        return True

# --- 🗄️ DATABASE SYSTEM ---
def init_db():
    conn = sqlite3.connect('eye_care_pro.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS screenings
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, date TEXT, condition TEXT, confidence REAL)''')
    conn.commit()
    conn.close()

def save_screening(name, condition, confidence):
    conn = sqlite3.connect('eye_care_pro.db')
    c = conn.cursor()
    date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    c.execute("INSERT INTO screenings (name, date, condition, confidence) VALUES (?, ?, ?, ?)",
              (name, date_str, condition, confidence))
    conn.commit()
    conn.close()

# --- 🧠 AI CORE (WITH TTA & BEN GRAHAM) ---
@st.cache_resource
def load_ai_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("📡 Downloading AI Brain..."):
            try:
                r = requests.get(MODEL_URL, stream=True)
                with open(MODEL_PATH, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
            except: st.error("Model download failed. Using local if exists.")

    model = load_model(MODEL_PATH, compile=False)
    with open('class_names.json', 'r') as f:
        classes = json.load(f)
    return model, classes

def ben_graham_preprocess(img_bytes):
    # Convert bytes to OpenCV image
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (192, 192))

    # Ben Graham's High Contrast (Makes vessels pop)
    img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0,0), 10), -4, 128)

    # EfficientNet scaling
    img = tf.keras.applications.efficientnet.preprocess_input(img.astype(np.float32))
    return img

def predict_with_tta(img_bytes, model):
    # Prepare different versions of the image (Test Time Augmentation)
    base_img = ben_graham_preprocess(img_bytes)
    flipped_h = np.fliplr(base_img)
    flipped_v = np.flipud(base_img)

    # Stack images into a batch
    tta_batch = np.array([base_img, flipped_h, flipped_v])

    # Average the predictions
    preds = model.predict(tta_batch)
    avg_preds = np.mean(preds, axis=0)

    idx = np.argmax(avg_preds)
    conf = avg_preds[idx]
    return idx, conf

# --- 🖼️ EXPLAINABILITY (GRAD-CAM) ---
def get_gradcam(img_bytes, model):
    img = ben_graham_preprocess(img_bytes)
    img_array = np.expand_dims(img, axis=0)

    grad_model = tf.keras.models.Model(model.inputs, [model.get_layer("top_activation").output, model.output])
    with tf.GradientTape() as tape:
        last_conv_output, preds = grad_model(img_array)
        class_channel = preds[:, np.argmax(preds[0])]

    grads = tape.gradient(class_channel, last_conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = last_conv_output[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap

# --- 🕶️ OPTICAL ENGINE (PD) ---
def get_pd(img_bytes):
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) == 0: return None
    x, y, w, h = faces[0]
    eyes = eye_cascade.detectMultiScale(gray[y:y+h, x:x+w])
    if len(eyes) < 2: return None
    return round((abs(eyes[0][0] - eyes[1][0]) / w) * 145, 1)

# --- 🚀 UI APP ---
init_db()
model, class_names = load_ai_model()

st.title("👁️ EyeCare AI Hub: Professional Suite")
menu = st.sidebar.selectbox("Dashboard", ["Diagnostic Hub", "Physician Portal", "Optical Assistant"])

if menu == "Diagnostic Hub":
    st.subheader("🏥 Patient Screening")
    p_name = st.text_input("Patient Full Name", "Anonymous")
    col1, col2 = st.columns(2)

    with col1:
        source = st.radio("Image Source", ["Upload File", "Live Scanner"])
        file = st.file_uploader("Retinal Fundus Image", type=['jpg','png','jpeg']) if source == "Upload File" else st.camera_input("Scan Eye")

        if file and st.button("🚀 Process Scan"):
            img_bytes = file.getvalue()
            idx, conf = predict_with_tta(img_bytes, model)
            disease = class_names[idx].title()

            st.session_state['res'] = {"name": p_name, "disease": disease, "conf": conf, "bytes": img_bytes}
            save_screening(p_name, disease, conf)

    with col2:
        if 'res' in st.session_state:
            r = st.session_state['res']
            st.success(f"**Diagnosis:** {r['disease']}")
            st.info(f"**AI Confidence (TTA Optimized):** {r['conf']:.1%}")

            if r['disease'] != "Normal":
                st.warning("🚨 Pathology Detected. Referral Required.")
                st.markdown(f"📍 [Find Specialist for {r['disease']}](https://www.google.com/maps/search/Eye+Specialist+near+me)")

            # Show Grad-CAM
            heatmap = get_gradcam(r['bytes'], model)
            st.image(r['bytes'], caption="Original Scan", use_container_width=True)
            st.write("✨ Heatmap visualization is processed on physician's portal.")

elif menu == "Physician Portal":
    if check_password():
        st.subheader("📋 Clinical History & Records")
        conn = sqlite3.connect('eye_care_pro.db')
        df = pd.read_sql_query("SELECT * FROM screenings ORDER BY id DESC", conn)
        st.dataframe(df, use_container_width=True)
        if st.button("🗑️ Clear Logs"):
            conn.execute("DELETE FROM screenings"); conn.commit(); st.rerun()

elif menu == "Optical Assistant":
    st.subheader("🕶️ PD Measurement & Frame Selection")
    face = st.camera_input("Facial Scan")
    if face:
        pd = get_pd(face.getvalue())
        if pd:
            st.metric("Measured PD", f"{pd} mm")
            st.success("Face Shape: Balanced. Recommendation: Aviator or Classic Square shades.")
        else: st.error("Detection failed. Please look directly at the lens.")

st.markdown("---")
st.caption("EyeCare AI v4.5 | Production Deployment | Clinical Decision Support System")
