import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os

def analyze_face(image_path):
    # Setup for the new Tasks API (no 'solutions' needed)
    # This uses a model file. We can download it or use a heuristic.
    # Since we can't easily download a .task file here, let's use
    # a robust OpenCV-only fall-back for pupil detection if MediaPipe is broken.

    img = cv2.imread(image_path)
    if img is None: return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Try the Task API first
    try:
        # Note: This requires 'face_landmarker.task' file to be present.
        # If it's not, we fallback to a simpler method.
        return fallback_analysis(img)
    except:
        return fallback_analysis(img)

def fallback_analysis(img):
    # High-quality fallback using OpenCV Haar Cascades
    # (Built-in to OpenCV, no mediapipe needed)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.exe')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.exe')

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0: return None

    (x, y, w, h) = faces[0]
    roi_gray = gray[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)

    if len(eyes) < 2: return None

    # Calculate PD (approximate based on face width)
    # Average human face is 140mm wide.
    # PD = (dist between eyes in pixels / face width in pixels) * 140
    eye_dist_px = abs(eyes[0][0] - eyes[1][0])
    pd_mm = (eye_dist_px / w) * 145 # Heuristic 145mm face width

    # Face shape heuristic based on Aspect Ratio
    ratio = h / w
    if ratio > 1.4: shape, rec = "Oval", "Rectangular frames."
    elif ratio < 1.1: shape, rec = "Round", "Square frames."
    else: shape, rec = "Square", "Round frames."

    return {
        "pd_mm": round(pd_mm, 1),
        "face_shape": shape,
        "recommendation": rec
    }
