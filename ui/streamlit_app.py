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

# --- 💅 STYLING ---
st.markdown("""
    <style>
    :root { --primary: #1a73e8; --secondary: #0d47a1; --background: #f8f9fa; }
    .main { background-color: var(--background); }
    .main-header { font-family: 'Helvetica Neue', sans-serif; color: var(--secondary); font-weight: 700; border-bottom: 2px solid var(--primary); padding-bottom: 10px; margin-bottom: 25px; }

    /* Mirror Flip for Camera */
    [data-testid="stCameraInput"] {
        border: 5px solid var(--primary);
        border-radius: 20px;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(26, 115, 232, 0.3);
    }
    [data-testid="stCameraInput"] video { transform: scaleX(-1); }

    .stMetric { background: white; padding: 20px; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); border-left: 8px solid var(--primary); }
    [data-testid="stSidebar"] { background-image: linear-gradient(#ffffff, #e3f2fd); }

    /* Catchy Buttons */
    .stButton>button {
        background-color: var(--primary);
        color: white;
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: var(--secondary);
        transform: translateY(-2px);
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
        st.info("📡 Downloading AI Brain (80MB)... This may take 1-2 minutes.")
        try:
            r = requests.get(MODEL_URL, stream=True)
            r.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
            st.success("✅ Download Complete.")
        except Exception as e:
            st.error(f"❌ Download Failed: {e}")
            return None, [], None

    try:
        # Resolve class names path
        c_path = 'class_names.json'
        if not os.path.exists(c_path): c_path = os.path.join(os.path.dirname(__file__), '..', 'class_names.json')
        if not os.path.exists(c_path): return None, [], None
        with open(c_path, 'r') as f: classes = json.load(f)

        model = load_model(MODEL_PATH, compile=False)

        # ELITE GRAD-CAM ENGINE (Bypasses Disconnection Errors)
        grad_model = None
        try:
            # Find the main feature extraction layer
            # We look for the last layer with a 4D output that is connected to the graph
            for layer in reversed(model.layers):
                try:
                    # Test if we can build a sub-model with this layer
                    temp_model = tf.keras.models.Model(model.inputs, [layer.output, model.output])
                    grad_model = temp_model
                    break # Success!
                except:
                    continue
        except:
            grad_model = None

        return model, classes, grad_model
    except Exception as e:
        st.error(f"❌ Brain Load Error: {e}")
        return None, [], None
    except Exception as e:
        st.error(f"❌ Brain Load Error: {e}")
        return None, [], None

# Load Cascades once
@st.cache_resource
def load_cascades():
    f = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    e = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    return f, e

def get_pd(img_bytes, face_cascade, eye_cascade):
    try:
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        h_orig, w_orig = img.shape[:2]
        img_res = cv2.resize(img, (800, int(800 * h_orig / w_orig)))
        gray = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)

        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
        if len(faces) == 0: return None, None

        x, y, w, h = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]

        # UI Overlay: Facial Recognition Simulation
        overlay = img_res.copy()
        # Draw bounding box with corner accents
        c_len = 30
        cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 255, 0), 1)
        # Corners
        cv2.line(overlay, (x, y), (x+c_len, y), (0, 255, 0), 4)
        cv2.line(overlay, (x, y), (x, y+c_len), (0, 255, 0), 4)
        cv2.line(overlay, (x+w-c_len, y), (x+w, y), (0, 255, 0), 4)
        cv2.line(overlay, (x+w, y), (x+w, y+c_len), (0, 255, 0), 4)

        roi_gray = gray[y : y + int(h * 0.6), x : x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.05, 6, minSize=(30, 30))

        if len(eyes) < 2: return None, overlay

        eyes = sorted(eyes, key=lambda e: e[0])
        e1, e2 = eyes[0], eyes[1]

        # Calculate pupil centers (approx)
        p1 = (x + e1[0] + e1[2]//2, y + e1[1] + e1[3]//2)
        p2 = (x + e2[0] + e2[2]//2, y + e2[1] + e2[3]//2)

        # Draw landmarks
        cv2.circle(overlay, p1, 5, (0, 0, 255), -1) # Left pupil
        cv2.circle(overlay, p2, 5, (0, 0, 255), -1) # Right pupil
        cv2.line(overlay, p1, p2, (255, 255, 0), 2) # PD line

        eye_dist_px = abs(p1[0] - p2[0])
        pd_mm = round((eye_dist_px / w) * 145, 1)

        # Face mask logic (Circular)
        mask = np.zeros_like(img_res); center = (x + w//2, y + h//2); radius = int(max(w, h) * 0.6)
        cv2.circle(mask, center, radius, (255, 255, 255), -1); circular_face = cv2.bitwise_and(overlay, mask)
        cv2.circle(circular_face, center, radius, (232, 115, 26), 3)

        # Data Overlay
        cv2.putText(circular_face, f"ID: BIO-{hashlib.sha1(os.urandom(2)).hexdigest()[:4].upper()}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return pd_mm, circular_face
    except: return None, None

def is_retinal_scan(img_bytes, face_cascade):
    # Detects if a clear human face is present (Stricter threshold to avoid false positives)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Extremely strict: 15 neighbors and larger minSize
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
            if model is None:
                st.error("🧠 AI Brain is still loading or offline. Please refresh in a moment.")
            else:
                img_bytes = file.getvalue()
                # Use a soft warning instead of a hard block to prevent false positives
                is_likely_face = not is_retinal_scan(img_bytes, face_cascade)

                if is_likely_face:
                    st.warning("⚠️ **Note:** The system detects potential facial features. Please ensure you are uploading an **Internal Retinal Scan** for accurate results.")

                with st.spinner("Analyzing..."):
                        nparr = np.frombuffer(img_bytes, np.uint8)
                        orig = cv2.imdecode(nparr, cv2.IMREAD_COLOR); orig_res = cv2.resize(orig, (224, 224))
                        input_arr = tf.keras.applications.efficientnet.preprocess_input(orig_res.astype(np.float32))
                        input_batch = np.expand_dims(input_arr, 0)

                        # Single-pass Diagnosis + Heatmap attempt
                        try:
                            if grad_model:
                                with tf.GradientTape() as tape:
                                    conv_output, preds = grad_model(input_batch)
                                    idx = np.argmax(preds[0]); loss = preds[:, idx]
                                grads = tape.gradient(loss, conv_output)
                                pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
                                heatmap = conv_output[0] @ pooled[..., tf.newaxis]
                                heatmap = np.maximum(tf.squeeze(heatmap), 0) / (np.max(heatmap) if np.max(heatmap) > 0 else 1)
                                heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
                                heatmap_color = cv2.resize(heatmap_color, (224, 224))
                                cam = cv2.addWeighted(orig_res, 0.6, heatmap_color, 0.4, 0)
                            else:
                                preds = model.predict(input_batch)
                                idx = np.argmax(preds[0]); cam = orig_res
                        except:
                            # Final fallback
                            preds = model.predict(input_batch)
                            idx = np.argmax(preds[0]); cam = orig_res

                        conf = float(preds[0][idx]); cond = class_names[idx].title()
                        st.session_state['res'] = {"cond": cond, "conf": conf, "cam": cam, "pid": p_name}
                        save_case(p_name, cond, conf)
    with col2:
        if 'res' in st.session_state:
            r = st.session_state['res']
            st.markdown(f"""<div style="background: white; padding: 20px; border-radius: 15px; border: 1px solid #eee; margin-bottom: 20px;"><p style="color: gray; margin: 0;">Diagnosis</p><h2 style="margin: 0; color: #1a73e8;">{r['cond']}</h2><p style="margin: 0; font-weight: bold; color: #28a745;">Confidence: {r['conf']:.1%}</p></div>""", unsafe_allow_html=True)
            st.image(r['cam'], caption="AI Explainability Map")
            if r['cond'] != "Normal":
                st.warning("🚨 Clinical Referral Required.")
                clinics = ["Dr. Agarwal's Eye Hospital", "Third Eyecare and Vision Centre", "Imprexions Eye Care", "Spectacular Optics Eye Care", "Advanced Eyecare", "St. Thomas Eye Hospital"]
                for clinic in clinics:
                    st.markdown(f"✅ **{clinic}** [🔍 Locate](https://www.google.com/maps/search/{clinic.replace(' ', '+')}+Accra)")

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
        st.dataframe(pd.read_sql('SELECT * FROM screenings ORDER BY id DESC', conn), use_container_width=True)
    else: st.warning("Authentication required.")

elif t['opt'] in menu:
    st.markdown(f"<h1 class='main-header'>🕶️ {t['opt']}</h1>", unsafe_allow_html=True)
    col1, col2 = st.columns([1.2, 1])
    with col1:
        method = st.radio("Acquisition Mode", ["Upload Scan", "Live Bio-Scanner"], horizontal=True)
        file = st.file_uploader("Selfie", type=['jpg','png']) if method == "Upload Scan" else st.camera_input("Face Scan")
        if file:
            with st.spinner("Analyzing biometric data..."):
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
    st.markdown("<h1 class='main-header'>🤝 Clinical Partner Registration</h1>", unsafe_allow_html=True)
    with st.form("reg_form"):
        name, contact, loc = st.text_input("Clinic Name"), st.text_input("Contact"), st.text_area("Location")
        if st.form_submit_button("Submit Application"):
            conn = sqlite3.connect('clinical_records.db'); conn.execute('INSERT INTO partners (shop_name, location, contact) VALUES (?,?,?)', (name, loc, contact)); conn.commit(); st.success("Registration Sent!")

elif "Feedback" in menu:
    st.markdown("<h1 class='main-header'>💬 User Feedback</h1>", unsafe_allow_html=True)
    with st.form("feedback_form"):
        name = st.text_input("Name"); rating = st.slider("Rating", 1, 10, 8); comment = st.text_area("Feedback")
        if st.form_submit_button("Submit"):
            conn = sqlite3.connect('clinical_records.db'); conn.execute('INSERT INTO feedback (user, rating, comment) VALUES (?,?,?)', (name, rating, comment)); conn.commit(); st.success("Thank you!")

st.markdown("---")
st.caption("EyeCare AI Business Suite v8.5 | Enterprise-Grade Solution")
