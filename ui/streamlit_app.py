import streamlit as st
import requests
import json
from PIL import Image
import io
from fpdf import FPDF
import datetime
import os
import pandas as pd
from face_utils import analyze_face
from database import init_db, save_screening, get_history

# Initialize database on start
init_db()

# Setup
API_URL = "http://127.0.0.1:8000/detect/"

def generate_pdf(res, name):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Medical Screening Report", ln=True, align='C')
    pdf.set_font("Arial", size=10)
    pdf.cell(200, 10, txt=f"Patient: {name} | Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(100, 10, txt="AI Preliminary Findings:", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.cell(100, 10, txt=f"Detected Condition: {res['disease'].replace('_', ' ').title()}", ln=True)
    pdf.cell(100, 10, txt=f"AI Confidence: {res['confidence']:.1%}", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", 'I', 8)
    pdf.multi_cell(0, 5, txt="Disclaimer: This is an automated screening result. Please consult a qualified ophthalmologist.")
    return pdf.output(dest='S').encode('latin-1')

st.set_page_config(page_title="EyeCare AI Pro", page_icon="👁️", layout="wide")

# Sidebar Navigation
st.sidebar.title("🏥 Clinic Dashboard")
app_mode = st.sidebar.selectbox("Navigate", ["Screening Hub", "Patient History", "Frame & PD Assistant"])

if app_mode == "Screening Hub":
    st.title("👁️ AI Retinal Screening")
    patient_name = st.text_input("Patient Name", "Anonymous")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📸 Image Acquisition")
        input_type = st.radio("Choose Input Method", ["Upload File", "Live Camera"])

        if input_type == "Upload File":
            uploaded_file = st.file_uploader("Select Retinal Scan", type=["jpg", "jpeg", "png"])
        else:
            uploaded_file = st.camera_input("Take a photo of the retinal scan")

        if uploaded_file and st.button("🚀 Analyze Now"):
            with st.spinner("Processing..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "image/jpeg")}
                try:
                    response = requests.post(API_URL, files=files)
                    if response.status_code == 200:
                        res = response.json()
                        st.session_state['last_result'] = res
                        # Save to database
                        save_screening(patient_name, res['disease'], res['confidence'], res.get('gradcam_url', ''))
                    else: st.error("API Connection Error")
                except Exception as e: st.error(f"Error: {e}")

    with col2:
        st.subheader("📊 Results & Diagnostics")
        if 'last_result' in st.session_state:
            res = st.session_state['last_result']
            st.success(f"Condition: {res['disease'].replace('_', ' ').title()}")
            st.progress(res['confidence'])

            if res.get('gradcam_url'):
                st.image(f"http://127.0.0.1:8000{res['gradcam_url']}", caption="AI Attention Map")

            # Referral logic
            if res['disease'] != 'normal':
                st.error("⚠️ Urgent: Clinical Consultation Recommended")
                st.markdown(f"[🔍 Find nearest Ophthalmologist for {res['disease'].title()}](https://www.google.com/maps/search/Ophthalmologist+near+me)")

            pdf_bytes = generate_pdf(res, patient_name)
            st.download_button("📥 Export Medical Report", data=pdf_bytes, file_name=f"Report_{patient_name}.pdf", mime="application/pdf")

elif app_mode == "Patient History":
    st.title("📋 Screening Logs")
    history = get_history()
    if history:
        df = pd.DataFrame(history, columns=["ID", "Name", "Date", "Condition", "Confidence", "Heatmap URL"])
        st.dataframe(df.drop(columns=["Heatmap URL"]), use_container_width=True)
    else:
        st.info("No records found yet.")

elif app_mode == "Frame & PD Assistant":
    st.title("🕶️ Digital Frame Assistant")
    face_file = st.camera_input("Scan Face for PD & Shapes")
    if face_file:
        temp_p = "temp_face.jpg"
        with open(temp_p, "wb") as f: f.write(face_file.getbuffer())
        data = analyze_face(temp_p)
        if data:
            st.metric("Pupillary Distance", f"{data['pd_mm']} mm")
            st.info(f"Shape: {data['face_shape']} | Rec: {data['recommendation']}")
        os.remove(temp_p)

st.markdown("---")
st.caption("EyeCare AI Hub v4.0 (Production)")
