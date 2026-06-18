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
import hashlib

# --- 🌍 LOCALIZATION & CONFIG ---
LANG = {
    "English": {
        "title": "EyeCare AI Hub Pro",
        "hub": "Clinical Screening", "portal": "Business & Admin", "opt": "Optical Assistant",
        "name": "Patient Name", "upload": "Upload Retinal Scan", "process": "Run Analysis",
        "roi": "Revenue Calculator", "partner": "Clinic Registration"
    }
}

st.set_page_config(page_title="EyeCare AI Pro", layout="wide", page_icon="👁️")

# --- 💅 PROFESSIONAL STYLING (Bio-Scanner Feel) ---
st.markdown("""
    <style>
    /* Circular Scanner Style for Camera */
    [data-testid="stCameraInput"] {
        border: 4px solid #1a73e8;
        border-radius: 50%;
        overflow: hidden;
        width: 350px !important;
        height: 350px !important;
        margin: 0 auto;
        box-shadow: 0 0 20px rgba(26, 115, 232, 0.4);
    }
    [data-testid="stCameraInput"] video {
        object-fit: cover;
    }
    .stMetric {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #1a73e8;
    }
    </style>
    """, unsafe_allow_html=True)

MODEL_PATH = "retinal_final_boss.h5"
MODEL_URL = "https://www.dropbox.com/scl/fi/ruipg8kbuu435c0l73rfp/retinal_disease_model_v2.h5?rlkey=alk3qd9neodv1dehflhej0fcy&st=48evd1oe&dl=1"

# --- 🗄️ DATABASE ---
def init_db():
    conn = sqlite3.connect('clinical_records.db')
    conn.execute('CREATE TABLE IF NOT EXISTS screenings (id INTEGER PRIMARY KEY AUTOINCREMENT, pid TEXT, date TEXT, condition TEXT, confidence REAL)')
    conn.execute('CREATE TABLE IF NOT EXISTS partners (id INTEGER PRIMARY KEY AUTOINCREMENT, shop_name TEXT, location TEXT, contact TEXT)')
    conn.commit()
    conn.close()

def save_case(pid, cond, conf):
    conn = sqlite3.connect('clinical_records.db')
    date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    conn.execute('INSERT INTO screenings (pid, date, condition, confidence) VALUES (?,?,?,?)', (pid, date, cond, conf))
    conn.commit(); conn.close()

# --- 🧠 AI ENGINE ---
@st.cache_resource
def load_clinical_brain():
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 10000:
        try:
            r = requests.get(MODEL_URL, stream=True)
            with open(MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
        except: pass
    try:
        model = load_model(MODEL_PATH, compile=False)
        with open('class_names.json', 'r') as f: classes = json.load(f)
        return model, classes
    except: return None, []

def get_gradcam(img_bytes, model):
    try:
        nparr = np.frombuffer(img_bytes, np.uint8)
        orig = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        orig = cv2.resize(orig, (224, 224))
        input_arr = tf.keras.applications.efficientnet.preprocess_input(orig.astype(np.float32))

        target_layer = None
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.Model):
                for sub_layer in reversed(layer.layers):
                    if len(sub_layer.output_shape) == 4:
                        target_layer = sub_layer; break
            elif len(layer.output_shape) == 4:
                target_layer = layer
            if target_layer: break

        grad_model = tf.keras.models.Model(model.inputs, [target_layer.output, model.output])
        with tf.GradientTape() as tape:
            conv, preds = grad_model(np.expand_dims(input_arr, 0))
            class_idx = np.argmax(preds[0])
            loss = preds[:, class_idx]

        grads = tape.gradient(loss, conv)
        pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = conv[0] @ pooled[..., tf.newaxis]
        heatmap = np.maximum(tf.squeeze(heatmap), 0) / (np.max(heatmap) if np.max(heatmap) > 0 else 1)
        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        heatmap_color = cv2.resize(heatmap_color, (224, 224))
        return cv2.addWeighted(orig, 0.6, heatmap_color, 0.4, 0)
    except: return cv2.resize(cv2.imdecode(np.frombuffer(img_bytes, np.uint8), 1), (224,224))

def get_pd(img_bytes):
    try:
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 1. ENHANCED PRE-PROCESSING
        h_orig, w_orig = img.shape[:2]
        img_res = cv2.resize(img, (800, int(800 * h_orig / w_orig)))
        gray = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
        if len(faces) == 0: faces = face_cascade.detectMultiScale(gray, 1.05, 3)
        if len(faces) == 0: return None, None

        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        x, y, w, h = faces[0]

        # CIRCULAR CROP FOR "BIO-SCAN" LOOK
        mask = np.zeros_like(img_res)
        center = (x + w//2, y + h//2)
        radius = int(max(w, h) * 0.6)
        cv2.circle(mask, center, radius, (255, 255, 255), -1)
        circular_face = cv2.bitwise_and(img_res, mask)

        # Add a digital "scanning" circle
        cv2.circle(circular_face, center, radius, (232, 115, 26), 5)

        roi_gray = gray[y : y + int(h * 0.6), x : x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.05, 6, minSize=(30, 30))
        if len(eyes) < 2: eyes = eye_cascade.detectMultiScale(roi_gray, 1.02, 3)
        if len(eyes) < 2: return None, circular_face

        eyes = sorted(eyes, key=lambda e: e[0])
        best_pair = (eyes[0], eyes[1])
        if len(eyes) > 2:
            min_y_diff = float('inf')
            for i in range(len(eyes)):
                for j in range(i + 1, len(eyes)):
                    y_diff = abs(eyes[i][1] - eyes[j][1])
                    if y_diff < min_y_diff:
                        min_y_diff = y_diff
                        best_pair = (eyes[i], eyes[j])

        e1, e2 = best_pair
        eye_dist_px = abs((e1[0] + e1[2]/2) - (e2[0] + e2[2]/2))
        pd_mm = (eye_dist_px / w) * 145

        return round(pd_mm, 1), circular_face
    except Exception as e:
        return None, None

# --- 🚀 UI LAUNCH ---
init_db()
model, class_names = load_clinical_brain()
t = LANG["English"]
menu = st.sidebar.radio("Navigation", [t['hub'], t['portal'], t['opt'], "Partner Registration"])

if menu == t['hub']:
    st.title("👁️ Retinal Diagnostic Suite")
    p_name = st.text_input(t['name'], "Patient #"+hashlib.sha1(os.urandom(4)).hexdigest()[:5])
    col1, col2 = st.columns(2)
    with col1:
        method = st.radio("Input Method", ["Upload Scan", "Live Camera"], horizontal=True)
        file = st.file_uploader(t['upload'], type=['jpg','png','jpeg']) if method == "Upload Scan" else st.camera_input("Scan Retina")

        if file and st.button(t['process']):
            img_bytes = file.getvalue()
            with st.spinner("Analyzing..."):
                img = cv2.resize(cv2.imdecode(np.frombuffer(img_bytes, np.uint8), 1), (224, 224))
                prep = tf.keras.applications.efficientnet.preprocess_input(img.astype(np.float32))
                preds = model.predict(np.expand_dims(prep, 0))
                idx = np.argmax(preds[0]); conf = float(preds[0][idx]); cond = class_names[idx].title()
                cam = get_gradcam(img_bytes, model)
                st.session_state['res'] = {"cond": cond, "conf": conf, "cam": cam, "pid": p_name}
                save_case(p_name, cond, conf)
    with col2:
        if 'res' in st.session_state:
            r = st.session_state['res']
            st.success(f"Diagnosis: {r['cond']} ({r['conf']:.1%})")
            st.image(r['cam'], caption="AI Explainability Map")
            if r['cond'] != "Normal":
                st.warning("🚨 Clinical Referral Required.")
                st.subheader("📍 Recommended Clinics in Accra")

                clinics = [
                    "Dr. Agarwal's Eye Hospital",
                    "Third Eyecare and Vision Centre",
                    "Imprexions Eye Care",
                    "Spectacular Optics Eye Care",
                    "Advanced Eyecare",
                    "St. Thomas Eye Hospital"
                ]

                for clinic in clinics:
                    map_url = f"https://www.google.com/maps/search/{clinic.replace(' ', '+')}+Accra"
                    st.markdown(f"✅ **{clinic}** [🔍 Locate on Map]({map_url})")

                st.info("Please contact one of these certified centers for a comprehensive examination.")

elif menu == t['portal']:
    if st.sidebar.text_input("Admin Key", type="password") == "doctor123":
        st.title("📊 Business Intelligence Dashboard")

        # ROI CALCULATOR
        st.subheader("💰 Revenue & ROI Calculator")
        col1, col2 = st.columns(2)
        with col1:
            patients = st.number_input("Average Patients per Month", 10, 5000, 100)
            cost_manual = st.number_input("Cost of Manual Screening ($)", 5, 200, 50)
            cost_ai = st.number_input("Monthly AI Subscription Cost ($)", 50, 1000, 200)

        total_manual = patients * cost_manual
        savings = total_manual - cost_ai
        with col2:
            st.metric("Potential Monthly Savings", f"${savings:,.2f}")
            st.metric("Annual Profit Increase", f"${savings*12:,.2f}")
            st.write("This tool demonstrates the financial value of adopting EyeCare AI Pro in your clinic.")

        st.markdown("---")
        st.subheader("📋 Patient Screening Logs")
        conn = sqlite3.connect('clinical_records.db')
        st.dataframe(pd.read_sql('SELECT * FROM screenings ORDER BY id DESC', conn), use_container_width=True)
    else: st.warning("Authentication required.")

elif menu == t['opt']:
    st.title("🕶️ Bio-Metric Optical Assistant")
    col1, col2 = st.columns([1.2, 1])
    with col1:
        method = st.radio("Acquisition Mode", ["Upload Scan", "Live Bio-Scanner"], horizontal=True)
        file = st.file_uploader("Selfie", type=['jpg','png']) if method == "Upload Scan" else st.camera_input("Face Scan")
        if file:
            img_bytes = file.getvalue()
            pd_val, scan_img = get_pd(img_bytes)
            if scan_img is not None:
                st.session_state['pd'] = pd_val
                st.session_state['scan_img'] = scan_img
            else:
                st.error("Bio-Metric Scan Failed: Face/Eyes not centered. Please try again.")
    with col2:
        if 'scan_img' in st.session_state:
            st.image(st.session_state['scan_img'], caption="Validated Bio-Scan", use_container_width=True)
        if 'pd' in st.session_state and st.session_state['pd'] is not None:
            st.metric("Detected PD", f"{st.session_state['pd']} mm")
            st.success("Analysis: Balanced Proportions.")
            st.info("Recommendation: Geometric or Aviator frames.")

elif menu == "Partner Registration":
    st.title("🤝 Become a Registered Optical Partner")
    st.write("Register your shop to receive priority referrals from the EyeCare AI network.")
    with st.form("reg_form"):
        name = st.text_input("Shop/Clinic Name")
        loc = st.text_input("Location (City/State)")
        contact = st.text_input("Professional Email/Phone")
        if st.form_submit_button("Submit Application"):
            conn = sqlite3.connect('clinical_records.db')
            conn.execute('INSERT INTO partners (shop_name, location, contact) VALUES (?,?,?)', (name, loc, contact))
            conn.commit(); conn.close()
            st.success("Registration Sent! Our team will verify your clinic shortly.")

st.markdown("---")
st.caption("EyeCare AI Business Suite v8.0 | Enterprise-Grade Solution")
