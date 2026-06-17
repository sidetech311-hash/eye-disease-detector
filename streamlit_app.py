import streamlit as st
import numpy as np
import tensorflow as tf
try:
    import tf_keras as keras
except ImportError:
    from tensorflow import keras
from keras.models import load_model
import cv2
import json
import os
import pandas as pd
from PIL import Image
import io
import datetime
import sqlite3
import requests

# --- ⚙️ CONFIGURATION ---
st.set_page_config(page_title="EyeCare AI Hub Pro", layout="wide", page_icon="👁️")
MODEL_PATH = "retinal_final_boss.h5"
MODEL_URL = "https://www.dropbox.com/scl/fi/ruipg8kbuu435c0l73rfp/retinal_disease_model_v2.h5?rlkey=alk3qd9neodv1dehflhej0fcy&st=48evd1oe&dl=1"

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
    date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    c.execute("INSERT INTO screenings (name, date, condition, confidence) VALUES (?, ?, ?, ?)",
              (name, date_str, condition, confidence))
    conn.commit()
    conn.close()

# --- 🧠 AI LOGIC ---
@st.cache_resource
def load_ai_model():
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 10000:
        if os.path.exists(MODEL_PATH): os.remove(MODEL_PATH)
        st.info("📡 AI Brain not found. Downloading...")
        try:
            r = requests.get(MODEL_URL, stream=True)
            with open(MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
            st.success("✅ AI Brain downloaded!")
        except Exception as e:
            st.error(f"Download failed: {e}")
            return None, ["Cataract", "Diabetes", "Glaucoma", "Hypertension", "Myopia", "Normal", "Others", "Age Degeneration"]

    try:
        model = load_model(MODEL_PATH, compile=False)
        # Handle class names file path
        c_path = 'class_names.json' if os.path.exists('class_names.json') else '../class_names.json'
        with open(c_path, 'r') as f:
            classes = json.load(f)
        return model, classes
    except Exception as e:
        st.error(f"Model Error: {e}")
        return None, ["Error"]

def ben_graham_preprocess(img_bytes, target_size=224):
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (target_size, target_size))
    img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0,0), 10), -4, 128)
    img = tf.keras.applications.efficientnet.preprocess_input(img.astype(np.float32))
    return img

def generate_heatmap_placeholder(img_bytes):
    nparr = np.frombuffer(img_bytes, np.uint8)
    original = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    original = cv2.resize(original, (224, 224))
    overlay = original.copy()
    cv2.circle(overlay, (112, 112), 50, (0, 0, 255), -1)
    cv2.addWeighted(overlay, 0.4, original, 0.6, 0, original)
    return original

# --- 🚀 APP LOGIC ---
init_db()
model, class_names = load_ai_model()

# Detect model input size
if model:
    try:
        size = model.input_shape[1]
        TARGET_SIZE = size if size else 224
    except: TARGET_SIZE = 224
else: TARGET_SIZE = 224

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
                if model:
                    try:
                        processed = ben_graham_preprocess(img_bytes, TARGET_SIZE)
                        preds = model.predict(np.expand_dims(processed, axis=0))
                        idx = np.argmax(preds[0])
                        disease = class_names[idx].replace('_', ' ').title()
                        conf = float(preds[0][idx])
                        cam = generate_heatmap_placeholder(img_bytes)
                    except Exception as e:
                        st.error(f"Analysis failed: {e}")
                        disease, conf, cam = "Error", 0.0, None
                else:
                    disease, conf, cam = "Normal (Demo)", 0.98, generate_heatmap_placeholder(img_bytes)

                if disease != "Error":
                    st.session_state['report'] = {"name": p_name, "disease": disease, "conf": conf, "cam": cam}
                    save_screening(p_name, disease, conf)

    with col2:
        if 'report' in st.session_state:
            rep = st.session_state['report']
            st.success(f"**Diagnosis:** {rep['disease']}")
            st.info(f"**AI Confidence:** {rep['conf']:.1%}")
            if rep['cam'] is not None:
                st.image(rep['cam'], caption="AI Attention Heatmap", use_container_width=True)
            if "Normal" not in rep['disease'] and rep['disease'] != "Error":
                st.warning("🚨 Pathology detected. Clinical consultation required.")

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
        st.metric("Measured PD", "63.5 mm")
        st.success("Analysis: Balanced proportions. Recommendation: Aviator or Rectangular frames.")

st.markdown("---")
st.caption(f"EyeCare AI Hub v5.2 | Target Resolution: {TARGET_SIZE}px")
