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
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av

# ── CONFIGURATION & ATTRIBUTES ────────────────────────────────────────────────
GOOGLE_DRIVE_FILE_ID = "1v6YP_WYMgnsoGbY0MLtSGNDeQNr-HPDW"
MODEL_PATH = "FairVision.pt"
AGE_GROUPS = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]

st.set_page_config(page_title="FairVision Dashboard", page_icon="👁️", layout="centered")

# ── MINIMAL DARK THEME COMPONENT CUSTOM CSS ───────────────────────────────────
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

/* Customizing the mode selector look */
div[data-testid="stRadio"] label {
    font-family: 'Space Mono', monospace !important;
    color: #e8e7f8 !important;
}
</style>
""", unsafe_allow_html=True)

# ── WEIGHTS DOWNLOAD AND ARCHITECTURE ASSEMBLY ───────────────────────────────
class FairVisionResNet(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()
        self.backbone = models.resnet50(weights=None)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.6), 
            nn.Linear(num_ftrs, num_classes)
        )
    def forward(self, x):
        return self.backbone(x)

@st.cache_resource
def load_fairvision_assets():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model weights from secure storage..."):
            gdown.download(f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}", MODEL_PATH, quiet=False)
            
    # Initialize network architecture
    model = FairVisionResNet(num_classes=9)
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    
    # Safely unpack model weights dictionary
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict({k.replace("module.", ""): v for k, v in state_dict.items()}, strict=False)
    model.eval()
    
    # Load optimized localized face classifier matrix
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return model, face_cascade

model, face_cascade = load_fairvision_assets()

# Standardized Torchvision Tensor Transforms matching dataset configuration
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ── CENTRALIZED INFERENCE ENGINE LOOKUP ───────────────────────────────────────
def process_cv2_frame(img):
    """Processes an incoming NumPy BGR matrix image, tracks faces, and overlays inference results."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    
    for (x, y, w, h) in faces:
        try:
            # ROI Isolation
            crop_bgr = img[y:y+h, x:x+w]
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            pil_crop = Image.fromarray(crop_rgb)
            
            # Predict
            with torch.no_grad():
                tensor_img = transform(pil_crop).unsqueeze(0)
                outputs = model(tensor_img)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                confidence, prediction = torch.max(probabilities, 0)
            
            label_text = AGE_GROUPS[prediction.item()]
            confidence_pct = confidence.item() * 100
            
            if confidence_pct >= 45.0:
                text_str = f"{label_text} ({confidence_pct:.1f}%)"
                accent_color = (255, 111, 124)  # BGR custom Purple/Indigo
                font_scale = 0.6
            else:
                text_str = "Can't detect you, come closer"
                accent_color = (100, 100, 230)  # BGR Warning Crimson
                font_scale = 0.45
            
            # Rendering localized bounding UI frames
            cv2.rectangle(img, (x, y), (x + w, y + h), accent_color, 3)
            cv2.rectangle(img, (x, y - 30), (x + w, y), accent_color, -1)
            cv2.putText(
                img, text_str, (x + 6, y - 8), 
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2, cv2.LINE_AA
            )
        except Exception:
            continue
    return img

# ── ASYNCHRONOUS WEBRTC STREAMING HOOK ────────────────────────────────────────
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    processed_img = process_cv2_frame(img)
    return av.VideoFrame.from_ndarray(processed_img, format="bgr24")

# ── USER INTERFACE DISPLAY ENTRYPOINT ─────────────────────────────────────────
st.markdown("""
<div class="fv-wordmark">Fair<em>Vision</em> Dashboard</div>
<div class="fv-tagline">Bias-Mitigated Age Estimation Engine • ResNet-50 Implementation</div>
<div class="fv-hr"></div>
""", unsafe_allow_html=True)

# Application Navigation Mode (Ensures operation regardless of restrictive enterprise firewalls)
pipeline_mode = st.radio(
    "Select System Input Pipeline Source:",
    ["Live Webcam Stream Tracking", "Static Document Image Upload"]
)

if pipeline_mode == "Live Webcam Stream Tracking":
    st.info("💡 Grant browser webcam permissions when prompted to initiate the live tracking loop.")
    
    # Highly robust array of public STUN endpoints to handle secure proxy routing
    resilient_rtc_config = RTCConfiguration({
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun2.l.google.com:19302"]},
            {"urls": ["stun:global.stun.twilio.com:3478?transport=udp"]}
        ]
    })
    
    webrtc_streamer(
        key="fairvision-runtime-stream",
        video_frame_callback=video_frame_callback,
        rtc_configuration=resilient_rtc_config,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

else:
    st.markdown("### Upload Target Image Frame")
    uploaded_image = st.file_uploader("Drop a face image to pass through the model matrix:", type=["jpg", "jpeg", "png"])
    
    if uploaded_image is not None:
        # Convert static file data array to standard processing array format
        raw_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        static_cv2_img = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
        
        with st.spinner("Executing model forward pass metrics..."):
            evaluated_img = process_cv2_frame(static_cv2_img)
            
        # Swap array channels back to RGB for rendering inside Streamlit's interface properly
        final_rgb_display = cv2.cvtColor(evaluated_img, cv2.COLOR_BGR2RGB)
        st.image(final_rgb_display, caption="FairVision Audited Result Frame", use_container_width=True)