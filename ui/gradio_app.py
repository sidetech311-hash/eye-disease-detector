# gradio_app.py
import gradio as gr
import requests
import json

API_URL = "http://127.0.0.1:8000/detect/"

def predict(image):
    if image is None:
        return "Please upload an image."
    try:
        # Gradio gives us a numpy array (H,W,3) in RGB
        # Convert to bytes
        from PIL import Image
        import io
        buf = io.BytesIO()
        Image.fromarray(image).save(buf, format="PNG")
        img_bytes = buf.getvalue()
        files = {"file": ("upload.png", img_bytes, "image/png")}
        r = requests.post(API_URL, files=files, timeout=10)
        r.raise_for_status()
        res = r.json()
        disease = res.get("disease", "unknown")
        conf  = res.get("confidence", 0.0)
        status = res.get("status", "error")
        if status != "success":
            return f"❌ API error: {res}"
        # nice HTML output
        if conf >= 0.7:
            bg = "#d4edda"; fg = "#155724"
        elif conf >= 0.5:
            bg = "#fff3cd"; fg = "#856404"
        else:
            bg = "#f8d7da"; fg = "#721c24"
        html = f"""
        <div style="
            background:{bg};
            color:{fg};
            padding:1rem;
            border-radius:8px;
            font-size:1.2rem;
            font-weight:600;
            text-align:center;">
            <strong>{disease.replace('_', ' ').title()}</strong><br>
            <span style="font-size:0.9rem;">{conf:.0%}</span>
        </div>
        """
        return html
    except Exception as e:
        return f"❌ Request failed: {e}"

# Optional: show a couple of example images from your dataset
example_images = [
    "diabetes/000001.jpg",
    "normal/000001.jpg",
    "cataract/000001.jpg",
]  # adjust paths if needed

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy", label="Fundus photo (JPG/PNG)"),
    outputs=gr.HTML(label="Result"),
    title="👁️ Retinal Disease Detector",
    description="Upload a fundus photo to get a fast screening result. "
                "Model: MobileNetV2 fine‑tuned on 8 retinal disease classes.",
    examples=example_images,
    theme="dark",
    allow_flagging="never",
    analytics=False,
)

if __name__ == "__main__":
    demo.launch()
