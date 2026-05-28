import gradio as gr
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import gdown
import os
import cv2
import numpy as np

# ── CONFIGURATION & ATTRIBUTES ────────────────────────────────────────────────
GOOGLE_DRIVE_FILE_ID = "1v6YP_WYMgnsoGbY0MLtSGNDeQNr-HPDW"
MODEL_PATH = "FairVision.pt"
AGE_GROUPS = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]

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

# Download weights if they do not exist
if not os.path.exists(MODEL_PATH):
    print("Downloading model weights from secure storage...")
    gdown.download(f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}", MODEL_PATH, quiet=False)

# Initialize network architecture globally
model = FairVisionResNet(num_classes=9)
checkpoint = torch.load(MODEL_PATH, map_location="cpu")
state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
model.load_state_dict({k.replace("module.", ""): v for k, v in state_dict.items()}, strict=False)
model.eval()

# Load facial boundary classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Standardized Tensor Transforms
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ── CENTRALIZED INFERENCE ENGINE LOOKUP ───────────────────────────────────────
def process_core_frame(frame):
    if frame is None:
        return None
        
    # Convert incoming PIL or NumPy RGB image to BGR array for OpenCV tracking
    img = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Track faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    
    for (x, y, w, h) in faces:
        try:
            # Region-of-Interest Isolation
            crop_bgr = img[y:y+h, x:x+w]
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            pil_crop = Image.fromarray(crop_rgb)
            
            # Predict Age Group
            with torch.no_grad():
                tensor_img = transform(pil_crop).unsqueeze(0)
                outputs = model(tensor_img)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                confidence, prediction = torch.max(probabilities, 0)
            
            label_text = AGE_GROUPS[prediction.item()]
            confidence_pct = confidence.item() * 100
            
            if confidence_pct >= 45.0:
                text_str = f"{label_text} ({confidence_pct:.1f}%)"
                accent_color = (124, 111, 255)  # RGB custom Purple/Indigo (#7c6fff)
            else:
                text_str = "Position face closer"
                accent_color = (230, 100, 100)  # RGB Warning Coral
            
            # Draw bounding box and prediction text (Gradio outputs natively in RGB format)
            cv2.rectangle(frame, (x, y), (x + w, y + h), accent_color, 4)
            cv2.rectangle(frame, (x, y - 35), (x + w, y), accent_color, -1)
            cv2.putText(
                frame, text_str, (x + 6, y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA
            )
        except Exception:
            continue
            
    return frame

# ── GRADIO INTERFACE LAYOUT ───────────────────────────────────────────────────
with gr.Blocks(css="footer {visibility: hidden !important;}") as demo:
    gr.Markdown("<h1 style='text-align: center; color: #7c6fff;'>FairVision Real-Time Engine</h1>")
    gr.Markdown("<p style='text-align: center;'>Bias-Mitigated Age Estimation Engine • ResNet-50 Implementation</p>")
    
    with gr.Tabs():
        # ── TAB 1: LIVE WEBCAM STREAMING ──────────────────────────────────────
        with gr.TabItem("🎥 Live Continuous Tracking"):
            with gr.Row():
                # To clean up the interface, we can remove additional recording UI items 
                # by explicitly setting the input component properties
                webcam_input = gr.Image(
                    sources=["webcam"], 
                    streaming=True, 
                    label="Live Input Feed", 
                    show_download_button=False
                )
                video_output = gr.Image(label="Model Real-Time Output", show_download_button=False)
                
            webcam_input.stream(
                fn=process_core_frame, 
                inputs=webcam_input, 
                outputs=video_output, 
                stream_every=0.1,
                concurrency_limit=5
            )
            
        # ── TAB 2: STATIC FILE UPLOAD ─────────────────────────────────────────
        with gr.TabItem("📁 Static File Upload"):
            with gr.Row():
                static_input = gr.Image(type="numpy", label="Upload Photo (JPG / PNG)")
                static_output = gr.Image(label="Analyzed Result Output")
                
            # Triggers a processing sweep immediately when an image is loaded or updated
            static_input.change(
                fn=process_core_frame,
                inputs=static_input,
                outputs=static_output
            )

if __name__ == "__main__":
    demo.launch()