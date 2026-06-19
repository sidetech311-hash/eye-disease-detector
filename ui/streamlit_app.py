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

# --- 💅 PROFESSIONAL STYLING ---
st.markdown("""
    <style>
    :root { --primary: #1a73e8; --secondary: #0d47a1; --background: #f8f9fa; }
    .main { background-color: var(--background); }
    .main-header { font-family: 'Helvetica Neue', sans-serif; color: var(--secondary); font-weight: 700; border-bottom: 2px solid var(--primary); padding-bottom: 10px; margin-bottom: 25px; }

    /* Hardware-Safe Camera Styling */
    [data-testid="stCameraInput"] {
        border: 5px solid var(--primary);
        border-radius: 50% !important;
        overflow: hidden !important;
        box-shadow: 0 10px 30px rgba(26, 115, 232, 0.3);
        width: 350px !important;
        height: 350px !important;
        margin: 0 auto;
        position: relative;
    }
    [data-testid="stCameraInput"] > div { border-radius: 50% !important; overflow: hidden !important; }
    [data-testid="stCameraInput"] video {
        transform: scaleX(-1);
        object-fit: cover;
        border-radius: 50% !important;
    }
    /* Pulse and Scan Line */
    [data-testid="stCameraInput"]::after {
        content: ""; position: absolute; top: 0; left: 0; right: 0; bottom: 0;
        border: 2px solid rgba(26, 115, 232, 0.5); border-radius: 50%;
        animation: pulse 2s infinite; pointer-events: none;
    }
    [data-testid="stCameraInput"]::before {
        content: ""; position: absolute; top: 0; left: 0; width: 100%; height: 2px;
        background: rgba(232, 115, 26, 0.6); box-shadow: 0 0 15px rgba(232, 115, 26, 0.8);
        animation: scan 3s linear infinite; z-index: 10; pointer-events: none;
    }
    @keyframes scan { 0% { top: 0%; } 100% { top: 100%; } }
    @keyframes pulse { 0% { transform: scale(1); opacity: 1; } 100% { transform: scale(1.1); opacity: 0; } }

    .stMetric { background: white; padding: 20px; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); border-left: 8px solid var(--primary); }
    [data-testid="stSidebar"] { background-image: linear-gradient(#ffffff, #e3f2fd); }
    .stButton>button { background-color: var(--primary); color: white; border-radius: 10px; font-weight: 600; transition: all 0.3s; }
    .stButton>button:hover { background-color: var(--secondary); transform: translateY(-2px); }
    </style>
    """, unsafe_allow_html=True)

MODEL_PATH = "retinal_final_boss.h5"
MODEL_URL = "https://www.dropbox.com/scl/fi/ruipg8kbuu435c0l73rfp/retinal_disease_model_v2.h5?rlkey=alk3qd9neodv1dehflhej0fcy&st=48evd1oe&dl=1"

# --- 🗄️ DATABASE ---
def init_db():
    conn = sqlite3.connect('clinical_records.db')
    conn.execute('CREATE TABLE IF NOT EXISTS screenings (id INTEGER PRIMARY KEY AUTOINCREMENT, pid TEXT, date TEXT, condition TEXT, confidence REAL)')
    conn.execute('CREATE TABLE IF NOT EXISTS partners (id INTEGER PRIMARY KEY AUTOINCREMENT, shop_name TEXT, location TEXT, contact TEXT)')
    conn.execute('CREATE TABLE IF NOT EXISTS feedback (id INTEGER PRIMARY KEY AUTOINCREMENT, user TEXT, rating INTEGER, comment TEXT)')
    conn.commit(); conn.close()

def save_case(pid, cond, conf):
    conn = sqlite3.connect('clinical_records.db')
    date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    conn.execute('INSERT INTO screenings (pid, date, condition, confidence) VALUES (?,?,?,?)', (pid, date, cond, conf))
    conn.commit(); conn.close()

# --- 🧠 AI ENGINE ---
@st.cache_resource
def load_clinical_brain():
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 10000:
        if os.path.exists(MODEL_PATH): os.remove(MODEL_PATH)
        try:
            r = requests.get(MODEL_URL, stream=True)
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            block_size = 1024 * 8
            progress_bar = st.progress(0)
            downloaded = 0
            with open(MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=block_size):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress_bar.progress(min(downloaded / total_size, 1.0))
            st.success("✅ AI Brain Downloaded.")
        except Exception as e:
            st.error(f"Download Error: {e}")
            return None, [], None

    try:
        # Resolve class names
        c_path = 'class_names.json'
        if not os.path.exists(c_path): c_path = os.path.join(os.path.dirname(__file__), '..', 'class_names.json')
        if not os.path.exists(c_path): return None, [], None
        with open(c_path, 'r') as f: classes = json.load(f)

        # Load model using standard TF loader (will be Keras 2 after environment update)
        model = load_model(MODEL_PATH, compile=False)

        # Build Grad-CAM model
        grad_model = None
        for layer in reversed(model.layers):
            try:
                grad_model = tf.keras.models.Model(model.inputs, [layer.output, model.output])
                break
            except: continue

        return model, classes, grad_model
    except Exception as e:
        st.error(f"🧠 Brain Load Failure: {e}")
        if os.path.exists(MODEL_PATH): os.remove(MODEL_PATH)
        return None, [], None

@st.cache_resource
def load_cascades():
    f = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    e = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    return f, e

def ben_graham_process(img):
    img_res = cv2.resize(img, (224, 224))
    return cv2.addWeighted(img_res, 4, cv2.GaussianBlur(img_res, (0,0), 10), -4, 128)

def get_pd(img_bytes, face_cascade, eye_cascade):
    try:
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        h_orig, w_orig = img.shape[:2]
        # Standardize for consistent detection
        img_res = cv2.resize(img, (800, int(800 * h_orig / w_orig)))
        gray = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY)
        # Advanced Histogram Equalization (CLAHE)
        gray = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(gray)

        # Multi-pass face detection
        faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(80, 80))
        if len(faces) == 0:
            # Pass 2: More sensitive
            faces = face_cascade.detectMultiScale(gray, 1.05, 3)

        if len(faces) == 0: return None, None

        x, y, w, h = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]

        # Biometric HUD Overlay
        overlay = img_res.copy()
        c_len = int(w * 0.1); thick = 4; color = (232, 115, 26)
        for dx, dy in [(0,0), (1,0), (0,1), (1,1)]:
            px = x + dx * w; py = y + dy * h
            cv2.line(overlay, (px, py), (px + (1-2*dx)*c_len, py), color, thick)
            cv2.line(overlay, (px, py), (px, py + (1-2*dy)*c_len), color, thick)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 255, 255), 1)

        # Eye ROI (Upper 60%)
        roi_gray = gray[y : y + int(h * 0.6), x : x + w]
        # Multi-pass eye detection
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.05, 5, minSize=(25, 25))
        if len(eyes) < 2:
            # Pass 2: More sensitive eyes
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.03, 3)

        mask = np.zeros_like(img_res); center = (x + w//2, y + h//2); radius = int(max(w, h) * 0.65)
        cv2.circle(mask, center, radius, (255, 255, 255), -1)

        if len(eyes) < 2:
            # Found face but eyes failed - show the face box
            circular_face = cv2.bitwise_and(overlay, mask)
            cv2.circle(circular_face, center, radius, (26, 115, 232), 4)
            return None, circular_face

        eyes = sorted(eyes, key=lambda e: e[0])
        p1 = (x + eyes[0][0] + eyes[0][2]//2, y + eyes[0][1] + eyes[0][3]//2)
        p2 = (x + eyes[1][0] + eyes[1][2]//2, y + eyes[1][1] + eyes[1][3]//2)
        for p in [p1, p2]:
            cv2.drawMarker(overlay, p, (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
            cv2.circle(overlay, p, 5, (0, 0, 255), -1)
        cv2.line(overlay, p1, p2, (232, 115, 26), 2, cv2.LINE_AA)

        pd_mm = round((abs(p1[0] - p2[0]) / w) * 145, 1)
        circular_face = cv2.bitwise_and(overlay, mask)
        cv2.circle(circular_face, center, radius, (132, 232, 26), 4) # Success Green
        return pd_mm, circular_face
    except: return None, None
    except: return None, None

def is_retinal_scan(img_bytes, face_cascade):
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 15, minSize=(int(img.shape[0]*0.3), int(img.shape[0]*0.3)))
    return len(faces) == 0

# --- 🚀 UI LAUNCH ---
init_db()
model, class_names, grad_model = load_clinical_brain()
face_cascade, eye_cascade = load_cascades()
t = LANG["English"]

st.sidebar.markdown(f"<h2 style='text-align: center; color: #1a73e8;'>👁️ EyeCare AI</h2>", unsafe_allow_html=True)
st.sidebar.markdown("---")
menu = st.sidebar.radio("Navigation", [f"📋 {t['hub']}", f"📊 {t['portal']}", f"🕶️ {t['opt']}", "🤝 Partner Registration", "💬 User Feedback"])

if t['hub'] in menu:
    st.markdown(f"<h1 class='main-header'>🔬 {t['title']}</h1>", unsafe_allow_html=True)
    p_name = st.text_input(t['name'], "Patient #"+hashlib.sha1(os.urandom(4)).hexdigest()[:5])
    col1, col2 = st.columns([1, 1.2])
    with col1:
        method = st.radio("Input Method", ["Upload Scan", "Live Camera"], horizontal=True)
        file = st.file_uploader(t['upload'], type=['jpg','png','jpeg']) if method == "Upload Scan" else st.camera_input("Scan Retina")
        if file and st.button(t['process'], use_container_width=True):
            if model is None: st.error("🧠 AI Brain is loading or offline. Please refresh.")
            else:
                img_bytes = file.getvalue()
                if not is_retinal_scan(img_bytes, face_cascade): st.warning("⚠️ Retinal scan expected, detected face features.")
                with st.spinner("Analyzing..."):
                    nparr = np.frombuffer(img_bytes, np.uint8)
                    orig = cv2.imdecode(nparr, cv2.IMREAD_COLOR); orig_res = cv2.resize(orig, (224, 224))
                    enhanced = ben_graham_process(orig)
                    input_batch = np.expand_dims(tf.keras.applications.efficientnet.preprocess_input(orig_res.astype(np.float32)), 0)
                    try:
                        if grad_model:
                            with tf.GradientTape() as tape:
                                conv_output, preds = grad_model(input_batch)
                                idx = np.argmax(preds[0]); loss = preds[:, idx]
                            grads = tape.gradient(loss, conv_output)
                            if len(grads.shape) == 4:
                                pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
                                heatmap = np.maximum(tf.squeeze(conv_output[0] @ pooled[..., tf.newaxis]), 0)
                            else: heatmap = np.zeros((conv_output.shape[1], conv_output.shape[2]))
                            heatmap /= (np.max(heatmap) if np.max(heatmap) > 0 else 1)
                            cam = cv2.addWeighted(orig_res, 0.6, cv2.resize(cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET), (224,224)), 0.4, 0)
                        else:
                            preds = model.predict(input_batch); idx = np.argmax(preds[0]); cam = orig_res
                    except:
                        preds = model.predict(input_batch); idx = np.argmax(preds[0]); cam = orig_res
                    conf = float(preds[0][idx]); cond = class_names[idx].title()
                    st.session_state['res'] = {"cond": cond, "conf": conf, "cam": cam, "enhanced": enhanced, "pid": p_name}
                    save_case(p_name, cond, conf)
    with col2:
        if 'res' in st.session_state:
            r = st.session_state['res']
            st.markdown(f"""<div style="background: white; padding: 20px; border-radius: 15px; border: 1px solid #eee; margin-bottom: 20px;"><p style="color: gray; margin: 0;">Diagnosis</p><h2 style="margin: 0; color: #1a73e8;">{r['cond']}</h2><p style="margin: 0; font-weight: bold; color: #28a745;">Confidence: {r['conf']:.1%}</p></div>""", unsafe_allow_html=True)
            t1, t2, t3 = st.tabs(["AI Heatmap", "Enhanced View", "Raw Scan"])
            t1.image(r['cam'], use_container_width=True)
            t2.image(r['enhanced'], caption="Vessel Contrast Enhancement", use_container_width=True)
            t3.image(r['enhanced'], caption="Original Clinical Data", use_container_width=True)
            if r['cond'] != "Normal":
                st.warning("🚨 Clinical Referral Required.")
                clinics = ["Dr. Agarwal's Eye Hospital", "St. Thomas Eye Hospital"]
                for c in clinics: st.markdown(f"✅ **{c}** [🔍 Locate](https://www.google.com/maps/search/{c.replace(' ', '+')}+Accra)")

elif t['portal'] in menu:
    st.markdown(f"<h1 class='main-header'>📊 {t['portal']}</h1>", unsafe_allow_html=True)
    if st.sidebar.text_input("Admin Key", type="password") == "doctor123":
        patients = st.sidebar.number_input("Average Patients/Month", 10, 5000, 100)
        cost_manual = st.sidebar.number_input("Manual Cost ($)", 5, 200, 50)
        cost_ai = st.sidebar.number_input("AI Subscription ($)", 50, 1000, 200)
        savings = (patients * cost_manual) - cost_ai
        col1, col2 = st.columns(2)
        col1.metric("Monthly Savings", f"${savings:,.2f}")
        col2.metric("Annual Profit", f"${savings*12:,.2f}")
        st.markdown("---")
        conn = sqlite3.connect('clinical_records.db')
        df = pd.read_sql('SELECT * FROM screenings ORDER BY id DESC', conn)
        if not df.empty:
            c1, c2 = st.columns([1.5, 1])
            c1.dataframe(df, use_container_width=True)
            c2.write("**Disease Distribution**"); c2.bar_chart(df['condition'].value_counts())
        else: st.info("No records yet.")
    else: st.warning("Authentication required.")

elif t['opt'] in menu:
    st.markdown(f"<h1 class='main-header'>🕶️ {t['opt']}</h1>", unsafe_allow_html=True)
    col1, col2 = st.columns([1.2, 1])
    with col1:
        method = st.radio("Acquisition Mode", ["Upload Scan", "Live Bio-Scanner"], horizontal=True)
        file = st.file_uploader("Selfie", type=['jpg','png']) if method == "Upload Scan" else st.camera_input("Face Scan")
        if file:
            with st.spinner("Analyzing..."):
                pd_val, scan_img = get_pd(file.getvalue(), face_cascade, eye_cascade)
                if scan_img is not None: st.session_state['pd'], st.session_state['scan_img'] = pd_val, scan_img
                else: st.error("Capture Failed: Please look directly into the camera.")
    with col2:
        if 'scan_img' in st.session_state:
            st.image(st.session_state['scan_img'], use_container_width=True)
            if 'pd' in st.session_state and st.session_state['pd'] is not None:
                st.metric("Detected PD", f"{st.session_state['pd']} mm")
                st.info("Recommendation: Geometric or Aviator frames.")

elif "Partner" in menu:
    st.markdown("<h1 class='main-header'>🤝 Partner Registration</h1>", unsafe_allow_html=True)
    with st.form("reg_form"):
        name, contact, loc = st.text_input("Clinic Name"), st.text_input("Contact"), st.text_area("Location")
        if st.form_submit_button("Submit"):
            conn = sqlite3.connect('clinical_records.db'); conn.execute('INSERT INTO partners (shop_name, location, contact) VALUES (?,?,?)', (name, loc, contact)); conn.commit(); st.success("Registration Sent!")

elif "Feedback" in menu:
    st.markdown("<h1 class='main-header'>💬 User Feedback</h1>", unsafe_allow_html=True)
    with st.form("feedback_form"):
        name = st.text_input("Name"); rating = st.slider("Rating", 1, 10, 8); comment = st.text_area("Feedback")
        if st.form_submit_button("Submit"):
            conn = sqlite3.connect('clinical_records.db'); conn.execute('INSERT INTO feedback (user, rating, comment) VALUES (?,?,?)', (name, rating, comment)); conn.commit(); st.success("Thank you!")

st.markdown("---")
st.caption("EyeCare AI Business Suite v8.5 | Enterprise-Grade Solution")
