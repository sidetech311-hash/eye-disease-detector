import cv2
import mediapipe as mp
import numpy as np

# Standard MediaPipe initialization
mp_face_mesh = mp.solutions.face_mesh

def analyze_face(image_path):
    # Initialize MediaPipe
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5) as face_mesh:

        image = cv2.imread(image_path)
        if image is None: return None

        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.multi_face_landmarks:
            return None

        face_landmarks = results.multi_face_landmarks[0].landmark
        h, w, _ = image.shape

        # Pupil centers
        left_pupil = face_landmarks[468]
        right_pupil = face_landmarks[473]

        # Calculate pixel distance
        px_dist = np.sqrt((left_pupil.x - right_pupil.x)**2 + (left_pupil.y - right_pupil.y)**2) * w

        # Heuristic calibration (Iris approx 11.7mm)
        iris_width_px = np.sqrt((face_landmarks[469].x - face_landmarks[471].x)**2 +
                                (face_landmarks[469].y - face_landmarks[471].y)**2) * w
        mm_per_px = 11.7 / (iris_width_px if iris_width_px > 0 else 1)
        pd_mm = px_dist * mm_per_px

        # Face Shape Analysis
        forehead_w = abs(face_landmarks[103].x - face_landmarks[332].x)
        cheekbone_w = abs(face_landmarks[234].x - face_landmarks[454].x)
        jaw_w = abs(face_landmarks[172].x - face_landmarks[397].x)
        face_len = abs(face_landmarks[10].y - face_landmarks[152].y)

        if face_len > cheekbone_w * 1.5:
            shape, rec = "Oval", "Rectangular frames. Try deep tints like Tortoise."
        elif cheekbone_w > forehead_w and cheekbone_w > jaw_w:
            shape, rec = "Round", "Square frames. Bold colors like Black."
        elif abs(forehead_w - jaw_w) < 0.1:
            shape, rec = "Square", "Round frames. Light shades like Silver or Gold."
        else:
            shape, rec = "Heart", "Aviator frames. Soft colors like Rose Gold."

        return {"pd_mm": round(pd_mm, 1), "face_shape": shape, "recommendation": rec}
