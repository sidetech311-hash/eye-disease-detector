# 👁️ EyeCare AI Hub: Professional Clinical Suite

An advanced, end-to-end medical screening application that uses Deep Learning to detect retinal diseases, provide explainable AI insights (Grad-CAM), and assist with optical measurements.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge.svg)](https://eye-disease-detector-4zxefrkbe9jy5divsare6z.streamlit.app)

## 🚀 Key Features

### 1. 🏥 Diagnostic Hub
- **AI Screening:** Instant detection of 8 retinal conditions (Cataract, Diabetes, Glaucoma, Hypertension, Myopia, Age Degeneration, etc.).
- **Explainable AI (Grad-CAM):** Generates heatmaps to show exactly which areas of the fundus scan influenced the AI's decision.
- **Ben Graham Preprocessing:** High-contrast image enhancement to make micro-vascular abnormalities visible.
- **Clinical Reports:** One-click generation of PDF medical reports for patient records.

### 2. 🔐 Physician Portal
- **Secure Access:** Protected by a password-gate (`doctor123`) to maintain patient privacy.
- **Clinical Database:** Persistent storage of all screening logs, including patient names, conditions, and confidence scores.
- **History Tracking:** Review past patient data to monitor disease progression.

### 3. 🕶️ Optical Assistant
- **PD Measurement:** Accurate Pupillary Distance measurement using facial landmark detection.
- **Frame Recommendation:** AI-driven advice for frame shapes based on detected facial geometry.

## 🛠️ Tech Stack
- **AI Brain:** TensorFlow / Keras (EfficientNetB0 architecture)
- **Frontend:** Streamlit
- **Image Processing:** OpenCV (CLAHE & Ben Graham contrast enhancement)
- **Data:** SQLite3
- **Deployment:** Streamlit Cloud & GitHub

## 📦 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/sidetech311-hash/eye-disease-detector.git
   cd eye-disease-detector
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure the AI Model:**
   - Upload your `retinal_final_boss.h5` to Dropbox.
   - Get a direct download link (ending in `dl=1`).
   - Paste the link into `MODEL_URL` inside `streamlit_app.py`.

4. **Launch the app:**
   ```bash
   streamlit run streamlit_app.py
   ```

## ⚠️ Medical Disclaimer
This application is a **screening tool** and clinical decision support system. It is **not** a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of an ophthalmologist or other qualified health provider.

---
Developed as a professional clinical suite for retinal healthcare.
