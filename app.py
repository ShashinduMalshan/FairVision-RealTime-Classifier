import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import gdown
import os
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av  # Crucial for the modern WebRTC video processing framework

# ── CONFIG ────────────────────────────────────────────────────────────────────
GOOGLE_DRIVE_FILE_ID = "1v6YP_WYMgnsoGbY0MLtSGNDeQNr-HPDW"
MODEL_PATH = "FairVision.pt"
AGE_GROUPS = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]

st.set_page_config(page_title="FairVision Live Video", page_icon="👁️", layout="centered")

# ── MINIMAL DARK THEME CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;700&family=Space+Mono:wght@400;700&display=swap');

html, body, [class*="css"], .stApp, [data-testid="stAppViewContainer"],
[data-testid="stHeader"], [data-testid="stToolbar"], .main {
  background: #080810 !important;
  color: #c8c7e0 !important;
  font-family: 'Space Grotesk', sans-serif !important;
}
.block-container { padding: 2rem !important; max-width: 700px !important; }
#MainMenu, footer, header { visibility: hidden; }

.fv-wordmark {
  font-family: 'Space Mono', monospace;
  font-size: 2.2rem; font-weight: 700;
  color: #e8e7f8; letter-spacing: -0.04em; margin-bottom: 4px;
  text-align: center;
}
.fv-wordmark em { font-style: normal; color: #7c6fff; }
.fv-tagline { font-size: 0.82rem; color: #5a5878; margin-bottom: 1.5rem; text-align: center; }
.fv-hr { border: none; border-top: 1px solid #18182a; margin: 0 0 1.5rem; }
[data-testid="stSpinner"] p { color: #7c6fff !important; text-align: center; }
</style>
""", unsafe_allow_html=True)

# ── MODEL & ASSET LOADING ─────────────────────────────────────────────────────
class FairVisionResNet(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()
        self.backbone = models.resnet50(weights=None)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(nn.Dropout(0.6), nn.Linear(num_ftrs, num_classes))
    def forward(self, x):
        return self.backbone(x)

@st.cache_resource
def load_assets():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model weights..."):
            gdown.download(f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}", MODEL_PATH, quiet=False)
    model = FairVisionResNet(num_classes=9)
    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()}, strict=False)
    model.eval()
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return model, face_cascade

model, face_cascade = load_assets()

# PyTorch transformation logic
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ── REAL-TIME VIDEO PROCESSING ENGINE (UPDATED TO MODERN WEBRTC) ──────────────
class VideoInferenceProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = model
        self.face_cascade = face_cascade

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # Convert modern PyAV frame to numpy array (OpenCV format: BGR)
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the live stream frame
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
        
        for (x, y, w, h) in faces:
            try:
                # 1. Extract face coordinates and crop
                crop_bgr = img[y:y+h, x:x+w]
                
                # 2. Prepare crop for PyTorch (BGR array -> RGB PIL Image)
                crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                pil_crop = Image.fromarray(crop_rgb)
                
                # 3. Model Prediction
                with torch.no_grad():
                    tensor_img = transform(pil_crop).unsqueeze(0)
                    out = self.model(tensor_img)
                    probs = torch.nn.functional.softmax(out[0], dim=0)
                    conf, pred = torch.max(probs, 0)
                
                label = AGE_GROUPS[pred.item()]
                confidence_pct = conf.item() * 100
                
                # 4. Determine Text Output based on 45% Confidence Threshold
                if confidence_pct >= 45.0:
                    text_str = f"{label} ({confidence_pct:.1f}%)"
                    accent_color = (255, 111, 124)  # BGR Purple/Indigo (#7c6fff)
                    font_scale = 0.6
                else:
                    text_str = "Can't detect you, come closer"
                    accent_color = (100, 100, 230)  # BGR Warning Light Red/Coral
                    font_scale = 0.45 
                
                # 5. Draw bounding box and label overlay on frame
                cv2.rectangle(img, (x, y), (x + w, y + h), accent_color, 3)
                
                # Draw text background block directly above the box
                cv2.rectangle(img, (x, y - 30), (x + w, y), accent_color, -1)
                cv2.putText(
                    img, text_str, (x + 6, y - 8), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2, cv2.LINE_AA
                )
            except Exception as e:
                st.error(f"Error occurred while processing frame: {e}")
                continue
                
        # Return the modified frame wrapped back into a PyAV VideoFrame object
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ── UI LAYOUT ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="fv-wordmark">Fair<em>Vision</em> Video</div>
<div class="fv-tagline">Continuous Live Face Tracking & Age Inference</div>
<div class="fv-hr"></div>
""", unsafe_allow_html=True)

# WebRTC Stream configuration (uses free public STUN servers to access webcam in-browser)
rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

webrtc_streamer(
    key="fairvision-live-stream",
    video_processor_factory=VideoInferenceProcessor,  # Updated factory naming standard
    rtc_configuration=rtc_config,
    media_stream_constraints={"video": True, "audio": False},
)