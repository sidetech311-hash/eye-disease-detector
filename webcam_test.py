# START COPY
import cv2
import requests
import json
import time

# ----------------------------------------------------------------------
# 🔧 CONFIGURATION – EDIT ONLY THESE LINES IF NEEDED
# ----------------------------------------------------------------------
API_URL = "http://127.0.0.1:8000/detect/"      # ← Local API
# If you deployed to Render.com, replace the line above with:
# API_URL = "https://your-app.onrender.com/detect/"

# How often to send a frame (seconds). Lower = more responsive, higher = less load on API
SEND_INTERVAL = 0.5   # 2 frames per second – adjust to your liking
# ----------------------------------------------------------------------

def main():
    # Open the default webcam (0). If you have multiple cams, try 1, 2, …
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam")
        return

    print("✅ Webcam opened. Press 'q' to quit.")
    last_send = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to grab frame")
            break

        # Show the live feed
        cv2.imshow('Webcam – Press q to quit', frame)

        # Throttle how often we send to the API
        now = time.time()
        if now - last_send >= SEND_INTERVAL:
            last_send = now

            # Encode frame as JPEG (no need to save to disk)
            _, img_encoded = cv2.imencode('.jpg', frame)
            files = {'file': ('frame.jpg', img_encoded.tobytes(), 'image/jpeg')}

            try:
                resp = requests.post(API_URL, files=files, timeout=5)
                if resp.status_code == 200:
                    result = resp.json()
                    disease = result.get('disease', 'error')
                    conf  = result.get('confidence', 0)
                    status = result.get('status', 'error')

                    # Overlay result on the frame (optional, but nice)
                    label = f"{disease}: {conf:.0%}"
                    color = (0, 255, 0) if status == "success" else (0, 0, 255)
                    cv2.putText(frame, label, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                else:
                    print(f"⚠️ API error {resp.status_code}: {resp.text}")
            except requests.exceptions.RequestException as e:
                print(f"⚠️ Request failed: {e}")

        # Break on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("👋 Webcam released. Bye!")

if __name__ == "__main__":
    main()
# END COPY
