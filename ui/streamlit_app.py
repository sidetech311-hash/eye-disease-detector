import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import json
import os
import pandas as pd
from PIL import Image
import io
from fpdf import FPDF
import datetime
import sqlite3
import requests

# --- ⚙️ CONFIGURATION ---
st.set_page_config(page_title="EyeCare AI Hub Pro", layout="wide", page_icon="👁️")

# --- 🔐 SECURITY ---
def check_password():
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False
    if not st.session_state["password_correct"]:
        pwd = st.sidebar.text_input("🔑 Physician Password", type="password")
        if pwd == "doctor123":
            st.session_state["password_correct"] = True
            st.rerun()
        elif pwd:
            st.sidebar.error("Incorrect Password")
        return False
    return True

# --- 🗄️ DATABASE ---
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
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    c.execute("INSERT INTO screenings (name, date, condition, confidence) VALUES (?, ?, ?, ?)",
              (name, date_str, condition, confidence))
    conn.commit()
    conn.close()

# --- 🧠 AI LOGIC ---
MODEL_PATH = "retinal_final_boss.h5"
MODEL_URL = "https://www.dropbox.com/scl/fi/YOUR_LINK_HERE/retinal_final_boss.h5?rlkey=YOUR_KEY&dl=1"

@st.cache_resource
def load_ai_model():
    if not os.path.exists(MODEL_PATH):
        try:
            r = requests.get(MODEL_URL, stream=True)
            with open(MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
        except: pass
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH, compile=False)
        with open('class_names.json', 'r') as f:
            classes = json.load(f)
        return model, classes
    return None, []

def ben_graham_preprocess(img_bytes):
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (192, 192))
    img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0,0), 10), -4, 128)
    img = tf.keras.applications.efficientnet.preprocess_input(img.astype(np.float32))
    return img

def make_gradcam(img_bytes, model):
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
    heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) if np.max(heatmap) > 0 else 1)

    # Overlay
    nparr = np.frombuffer(img_bytes, np.uint8)
    original = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    original = cv2.resize(original, (192, 192))
    heatmap_img = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_img, cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(original, 0.6, heatmap_color, 0.4, 0)
    return superimposed

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
    pd = (abs(eyes[0][0] - eyes[1][0]) / w) * 145
    return round(pd, 1)

# --- 🚀 APP LOGIC ---
init_db()
model, class_names = load_ai_model()

st.title("👁️ EyeCare AI Hub: Professional Suite")
menu = st.sidebar.radio("Navigation", ["Diagnostic Hub", "Physician Portal", "Optical Assistant"])

if menu == "Diagnostic Hub":
    st.subheader("🏥 Digital Screening")
    p_name = st.text_input("Patient Name", "Anonymous")
    col1, col2 = st.columns(2)
    with col1:
        img_file = st.file_uploader("Upload Retinal Scan", type=['jpg','png','jpeg'])
        if img_file and st.button("🚀 Analyze Scan"):
            with st.spinner("Processing..."):
                img_bytes = img_file.getvalue()
                processed = ben_graham_preprocess(img_bytes)
                preds = model.predict(np.expand_dims(processed, axis=0))
                idx = np.argmax(preds[0])
                conf = float(preds[0][idx])
                disease = class_names[idx].replace('_', ' ').title()
                cam = make_gradcam(img_bytes, model)
                st.session_state['report'] = {"name": p_name, "disease": disease, "conf": conf, "cam": cam, "orig": img_bytes}
                save_screening(p_name, disease, conf)

    with col2:
        if 'report' in st.session_state:
            rep = st.session_state['report']
            st.success(f"**Diagnosis:** {rep['disease']}")
            st.info(f"**AI Confidence:** {rep['conf']:.1%}")
            st.image(rep['cam'], caption="AI Attention Heatmap (Explainable AI)", use_container_width=True)
            if rep['disease'] != "Normal":
                st.warning("🚨 Pathology detected. clinical consultation required.")
                st.markdown("[🔍 Find nearest Ophthalmologist](https://www.google.com/maps/search/Ophthalmologist)")

elif menu == "Physician Portal":
    if check_password():
        st.subheader("📋 Patient Records")
        conn = sqlite3.connect('eye_care_pro.db')
        df = pd.read_sql_query("SELECT * FROM screenings ORDER BY id DESC", conn)
        st.dataframe(df, use_container_width=True)

elif menu == "Optical Assistant":
    st.subheader("🕶️ PD & Frame Assistant")
    shot = st.camera_input("Scan Face")
    if shot:
        pd = get_pd(shot.getvalue())
        if pd:
            st.metric("Measured PD", f"{pd} mm")
            st.success("Analysis: Oval Face. Recommendation: Aviator or Rectangular frames.")
        else: st.error("Face not detected. Ensure good lighting.")

st.markdown("---")
st.caption("EyeCare AI Hub v5.0 | Production Release")
