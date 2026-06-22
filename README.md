# 👁️ EyeCare AI Hub Pro (Enterprise v10.1)

[![Clinical Validation](https://img.shields.io/badge/Status-WHO--Compliant--Ready-blue)](https://eyecare-ai.com)
[![License](https://img.shields.io/badge/Security-SHA--256--Anonymized-green)](https://github.com)

**EyeCare AI Hub Pro** is a disruptive Clinical Decision Support System (CDSS) designed for large-scale retinal screening in low-resource environments. It leverages Deep Learning to detect major ocular pathologies and provides a full suite of business analytics for hospital management.

## 🚀 Key Features
- **Multi-Disease Screening**: Instant detection of Cataract, Diabetes, Glaucoma, Hypertension, Myopia, and more.
- **Explainable AI (XAI)**: High-resolution Grad-CAM heatmaps for clinical auditability.
- **Edge-First Design**: Low-bandwidth mode with offline data buffering for rural outreach.
- **Biometric Hub**: AI-powered facial recognition for automated Pupillary Distance (PD) calibration.
- **Automated Reporting**: Instant generation of Clinical PDF Reports for patient records.

## 🛠️ Technical Stack
- **AI Core**: TensorFlow 2.15.0 / EfficientNet (B0-B7 Optimized)
- **Computer Vision**: OpenCV 4.x with multi-pass Haar Cascades
- **Backend**: Python 3.11 / SQLite3 / FastAPI
- **Frontend**: Streamlit Enterprise
- **Compliance**: SHA-256 data anonymization (GDPR/HIPAA Standard)

## 📦 Installation & Setup
```bash
git clone https://github.com/sidetech311-hash/eye-disease-detector.git
cd eye-disease-detector
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## 📊 Performance Audit
Run the validation suite to generate the formal WHO-submission report:
```bash
python clinical_validation.py
```

## 🤝 Partners & Referral Network
Currently integrated with major eye hospitals in Accra, Ghana including Dr. Agarwal's and St. Thomas Eye Hospital.

---
**Developer**: Hayford Kofi Quaye  
**Scientific Advisor**: [Clinical Partner Name]  
**Location**: Accra, Ghana
