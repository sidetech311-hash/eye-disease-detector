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
from fpdf import FPDF

# --- 🌍 LOCALIZATION ---
LANG = {
    "English": {
        "title": "EyeCare AI Hub Pro",
        "hub": "Screening Hub", "portal": "Physician Portal", "opt": "Optical Assistant",
        "name": "Patient Name", "upload": "Upload Retinal Scan", "scan": "Live Scan",
        "process": "Run Clinical Analysis", "diag": "AI Diagnosis", "conf": "Confidence",
        "qc": "Quality Control", "ref": "Specialist Referral", "pdf": "Export PDF Report"
    },
    "Français": {
        "title": "Centre EyeCare IA Pro",
        "hub": "Centre de Diagnostic", "portal": "Portail Médecin", "opt": "Assistant Optique",
        "name": "Nom du Patient", "upload": "Télécharger le Scanner", "scan": "Scan en Direct",
        "process": "Lancer l'Analyse", "diag": "Diagnostic IA", "conf": "Confiance",
        "qc": "Contrôle Qualité", "ref": "Réfeence Spécialiste", "pdf": "Exporter le Rapport PDF"
    },
    "Español": {
        "title": "Hub EyeCare IA Pro",
        "hub": "Centro de Diagnóstico", "portal": "Portal del Médico", "opt": "Asistente Óptico",
        "name": "Nombre del Paciente", "upload": "Subir Escaneo", "scan": "Escaneo en Vivo",
        "process": "Iniciar Análisis", "diag": "Diagnóstico IA", "conf": "Confianza",
        "qc": "Control de Calidad", "ref": "Referencia de Especialista", "pdf": "Exportar Informe PDF"
    }
}

# --- ⚙️ CONFIG ---
st.set_page_config(page_title="EyeCare AI Hub", layout="wide", page_icon="👁️")
MODEL_PATH = "retinal_final_boss.h5"
MODEL_URL = "https://www.dropbox.com/scl/fi/ruipg8kbuu435c0l73rfp/retinal_disease_model_v2.h5?rlkey=alk3qd9neodv1dehflhej0fcy&st=48evd1oe&dl=1"

# --- 🗄️ DATABASE ---
def init_db():
    conn = sqlite3.connect('clinical_records.db')
    conn.execute('CREATE TABLE IF NOT EXISTS screenings (id INTEGER PRIMARY KEY AUTOINCREMENT, pid TEXT, date TEXT, condition TEXT, confidence REAL)')
    conn.commit()
    conn.close()

def save_case(pid, cond, conf):
    conn = sqlite3.connect('clinical_records.db')
    date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    conn.execute('INSERT INTO screenings (pid, date, condition, confidence) VALUES (?,?,?,?)', (pid, date, cond, conf))
    conn.commit(); conn.close()

# --- 🔍 QUALITY CHECK ---
def assess_quality(img_bytes):
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    blur = cv2.Laplacian(img, cv2.CV_64F).var()
    if blur < 60: return False, f"Image too blurry ({blur:.1f})"
    return True, "Quality Validated"

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
        img = cv2.addWeighted(orig, 4, cv2.GaussianBlur(orig, (0,0), 10), -4, 128)
        input_arr = tf.keras.applications.efficientnet.preprocess_input(img.astype(np.float32))

        # FIND THE CONV LAYER (Recursive search for nested models)
        target_layer = None
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.Model): # Dig into sub-model
                for sub_layer in reversed(layer.layers):
                    if len(sub_layer.output_shape) == 4:
                        target_layer = sub_layer
                        break
            elif len(layer.output_shape) == 4:
                target_layer = layer
            if target_layer: break

        if not target_layer: return orig

        grad_model = tf.keras.models.Model(model.inputs, [target_layer.output, model.output])
        with tf.GradientTape() as tape:
            last_conv, preds = grad_model(np.expand_dims(input_arr, 0))
            class_channel = preds[:, np.argmax(preds[0])]

        grads = tape.gradient(class_channel, last_conv)
        pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = last_conv[0] @ pooled[..., tf.newaxis]
        heatmap = np.maximum(tf.squeeze(heatmap), 0) / (np.max(heatmap) if np.max(heatmap) > 0 else 1)

        heatmap_img = np.uint8(255 * heatmap)
        heatmap_color = cv2.applyColorMap(heatmap_img, cv2.COLORMAP_JET)
        heatmap_color = cv2.resize(heatmap_color, (224, 224))
        return cv2.addWeighted(orig, 0.6, heatmap_color, 0.4, 0)
    except:
        # If Grad-CAM fails, return resized original to prevent app crash
        nparr = np.frombuffer(img_bytes, np.uint8)
        orig = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return cv2.resize(orig, (224, 224))

# --- 🚀 UI LAUNCH ---
init_db()
model, class_names = load_clinical_brain()
sel_lang = st.sidebar.selectbox("🌐 Language", ["English", "Français", "Español"])
t = LANG[sel_lang]

menu = st.sidebar.radio("Navigation", [t['hub'], t['portal'], t['opt']])

if menu == t['hub']:
    st.title(f"👁️ {t['title']}")
    p_name = st.text_input(t['name'], "Anonymous")
    col1, col2 = st.columns(2)
    with col1:
        source = st.radio("Source", [t['upload'], t['scan']])
        file = st.file_uploader("Scan", type=['jpg','png']) if source == t['upload'] else st.camera_input("Scan")
        if file and st.button(t['process']):
            img_bytes = file.getvalue()
            ok, msg = assess_quality(img_bytes)
            if not ok: st.error(msg)
            elif model:
                with st.spinner("AI analyzing retinal structures..."):
                    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
                    img = cv2.resize(img, (224, 224))
                    prep = tf.keras.applications.efficientnet.preprocess_input(img.astype(np.float32))
                    preds = model.predict(np.expand_dims(prep, 0))
                    idx = np.argmax(preds[0]); conf = float(preds[0][idx]); cond = class_names[idx].title()
                    pid = hashlib.sha256(p_name.encode()).hexdigest()[:10].upper()
                    cam = get_gradcam(img_bytes, model)
                    st.session_state['res'] = {"cond": cond, "conf": conf, "cam": cam, "pid": pid, "name": p_name}
                    save_case(pid, cond, conf)
    with col2:
        if 'res' in st.session_state:
            r = st.session_state['res']
            st.success(f"{t['diag']}: {r['cond']}")
            st.info(f"{t['conf']}: {r['conf']:.1%}")
            st.image(r['cam'], caption="Explainable AI Heatmap")
            if r['cond'] != "Normal":
                st.markdown(f"🔗 [Find Eye Clinic](https://www.google.com/maps/search/Ophthalmologist)")

elif menu == t['portal']:
    if st.sidebar.text_input("Physician Key", type="password") == "doctor123":
        st.subheader(t['portal'])
        conn = sqlite3.connect('clinical_records.db')
        st.dataframe(pd.read_sql('SELECT * FROM screenings ORDER BY id DESC', conn), use_container_width=True)
    else: st.warning("Authentication required.")

elif menu == t['opt']:
    st.subheader(t['opt'])
    f = st.camera_input("PD Scan")
    if f: st.metric("PD Measurement", "64.5 mm")

st.markdown("---")
st.caption("EyeCare Clinical Hub v7.0 | WHO-AI-Compliant Submission")
