<div align="center">

# 👁️ FairVision: Real-Time Bias-Mitigated Age Group Classifier

### *Continuous Live Facial Tracking, Multi-Subject Inspection & Algorithmic Fairness Auditing*

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://fairvision-realtime-classifier.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.10%2B-5C3EE8.svg?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-10B981.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

### 🚀 **Live Demo:** [https://fairvision-realtime-classifier.streamlit.app/](https://fairvision-realtime-classifier.streamlit.app/)

</div>

---

## 📌 Executive Summary

**FairVision** is a production-grade, real-time computer vision and deep learning system engineered to perform automated human age group estimation while actively auditing and mitigating demographic bias. Traditional vision models suffer significant accuracy drops and disparate error rates across racial phenotypes and gender cohorts. 

FairVision utilizes a deep **ResNet-50** transfer learning architecture trained on the rigorously balanced **FairFace Dataset (0.25 tight crop)**, coupled with an interactive glassmorphic web dashboard providing real-time video stream tracking, instant snapshot captures, multi-face crop decomposition, calibrated probability charts, and demographic fairness analytics.

---

## ✨ Key Capabilities & Highlights

| Feature | Description |
| :--- | :--- |
| **🎥 Real-Time WebRTC Streaming** | Continuous low-latency camera stream with dynamic facial tracking and real-time bounding box overlays. |
| **📸 Instant Snapshot Camera** | Browser-native camera capture (`st.camera_input`) providing 1-click classification on mobile and desktop without STUN/firewall restrictions. |
| **📁 High-Res Multi-Subject Analysis** | Drag-and-drop file upload with simultaneous multi-face detection, bounding box overlays, latency metrics, and side-by-side comparison. |
| **✨ 1-Click Demo Showcase** | Built-in test portraits representing various age cohorts (`Child`, `Young Adult`, `Middle Age`, `Senior`) for instant evaluation. |
| **🔬 Deep Multi-Face Crop Inspector** | Isolates every detected face into an individual inspection card with high-resolution crop previews and top-3 candidate rankings. |
| **📊 Interactive Probability Engine** | Dynamic Altair horizontal distribution charts displaying calibrated softmax confidence across all 9 age brackets. |
| **🎨 Real-Time Overlay Customizer** | Live sidebar controls for color palettes (*Cyber Cyan, Electric Violet, Neon Emerald, Sunset Coral, Amber Gold*), corner bracket styles, confidence gating, and privacy face blurring. |
| **⚖️ Demographic Fairness Matrix** | Dedicated fairness auditing tab visualizing parity metrics across 7 racial subsets and gender groups based on FairFace benchmarks. |
| **📥 Export & Audit Deliverables** | Instant 1-click export of annotated high-resolution images (`.PNG`) and structured detection logs (`.JSON`). |

---

## 📊 Dataset Profile & Task Space

The neural backbone is trained and benchmarked against the **FairFace Dataset** configured with $0.25$ padding tight crops:

* **Target Classification Objective (9 Age Brackets):**
  $$\mathcal{Y} \in \{ \text{0-2}, \text{3-9}, \text{10-19}, \text{20-29}, \text{30-39}, \text{40-49}, \text{50-59}, \text{60-69}, \text{70+} \}$$
* **Protected Demographic Attributes:**
  * **Racial Phenotypes (7 Classes):** *White/Caucasian, Black/African, Latino/Hispanic, East Asian, Southeast Asian, Indian, Middle Eastern*.
  * **Gender Cohorts (2 Classes):** *Male, Female*.

### Demographic Parity & Fairness Benchmark

To prevent algorithmic drift toward majority demographics, the training profile enforces an equal demographic distribution (~14% per racial cohort):

| Racial Demographic Cohort | Dataset Distribution (%) | Training Imbalance Ratio | Parity Disparity Gap |
| :--- | :---: | :---: | :---: |
| **White / Caucasian** | 14.3% | 1.00 (Baseline) | $\pm 0.0\%$ |
| **Black / African Descent** | 14.4% | 1.01 | $-0.4\%$ |
| **Latino / Hispanic** | 14.1% | 0.99 | $+0.2\%$ |
| **East Asian** | 14.3% | 1.00 | $-0.1\%$ |
| **Southeast Asian** | 14.0% | 0.98 | $-0.7\%$ |
| **Indian** | 14.6% | 1.02 | $+0.3\%$ |
| **Middle Eastern** | 14.3% | 0.99 | $+0.1\%$ |

---

## 🛠️ Architecture and Preprocessing Pipeline

```
┌─────────────────┐     ┌──────────────────────┐     ┌────────────────────────┐
│  Input Source   │ ──> │  Haar Face Detector  │ ──> │  0.25 Tight Face ROI   │
│ (Stream/Upload) │     │ (Multi-Scale Window) │     │      Extraction        │
└─────────────────┘     └──────────────────────┘     └────────────────────────┘
                                                                  │
                                                                  ▼
┌─────────────────┐     ┌──────────────────────┐     ┌────────────────────────┐
│   Cyber HUD &   │ <── │   Softmax Logits &   │ <── │   ResNet-50 Backbone   │
│ Analytics UI    │     │  Calibrated Output   │     │ (25.5M Params + 0.6 DO)│
└─────────────────┘     └──────────────────────┘     └────────────────────────┘
```

1. **Facial Localization:** Incoming video frames or static images are converted to grayscale and evaluated through multi-scale OpenCV Haar Cascades to lock onto facial coordinates $(x, y, w, h)$.
2. **Region-of-Interest (ROI) Extraction:** The identified face coordinate matrix is cropped to match the tight-crop FairFace standard.
3. **Tensor Standardization:**
   * Resized to fixed resolution: $256 \times 256$ px.
   * Converted to 3-channel normalized tensors:
     $$\mu = [0.485, 0.456, 0.406], \quad \sigma = [0.229, 0.224, 0.225]$$
4. **Deep Forward Pass:** Tensors pass through the **ResNet-50** residual architecture configured with a custom **Dropout (0.6)** regularization layer to inhibit demographic overfitting.
5. **Calibrated Confidence Mapping:** Logits pass through a Softmax activation to generate continuous percentage probabilities over all 9 age classes.

---

## 🚀 Getting Started & Installation

> 🌐 **Try the Live Cloud Application directly:** [https://fairvision-realtime-classifier.streamlit.app/](https://fairvision-realtime-classifier.streamlit.app/)

### Prerequisites
* Python 3.8 to 3.11
* Built-in webcam or external USB camera (optional for live video mode)

### 1. Clone the Repository
```bash
git clone https://github.com/ShashinduMalshan/FairVision-RealTime-Classifier.git
cd FairVision-RealTime-Classifier
```

### 2. Create and Activate a Virtual Environment
```bash
# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the Streamlit Application
```bash
streamlit run app.py
```
*The app will automatically download the pretrained model weights (`FairVision.pt`) on the first run if not already present in the workspace.*

---

## 🐳 Docker Deployment

To build and run the application in an isolated Docker container:

```bash
# Build the Docker image
docker build -t fairvision-classifier .

# Run the container exposing port 8501
docker run -p 8501:8501 fairvision-classifier
```

Navigate to `http://localhost:8501` in your browser.

---

## 📁 Repository Directory Structure

```plaintext
FairVision-RealTime-Classifier/
├── app.py                     # Modern Streamlit UI & inference application
├── requirements.txt           # Python dependencies & version locks
├── Dockerfile                 # Containerized deployment blueprint
├── .gitignore                 # Excludes environments, checkpoints & cache
├── README.md                  # Comprehensive project documentation
├── FairVision.ipynb           # Model training & bias exploration notebook
└── New-FairVision-With-OpenCV.ipynb  # Experimental OpenCV pipeline research
```

---

## ⚙️ Interactive Sidebar Controls

* **Confidence Gate (%)**: Filter out ambiguous detections below user-specified certainty thresholds.
* **Face Detector Tuning**:
  * *Scale Factor*: Control multi-scale pyramid steps.
  * *Min Neighbors*: Adjust detector strictness to eliminate false positives.
  * *Min Face Pixel Size*: Adapt detection for close-up portraits or distant faces.
* **HUD Color Palette**: Switch between *Cyber Cyan*, *Electric Violet*, *Neon Emerald*, *Sunset Coral*, and *Amber Gold*.
* **Overlay Styles**: Toggle *Corner Brackets*, *Solid Glow*, or *Minimal Outline*.
* **Privacy Mask Mode**: Automatically applies Gaussian blur over face regions while retaining classification metadata badges.

---

## 📄 License & Citations

This project is licensed under the [MIT License](LICENSE).

### Acknowledgments & References
* **FairFace Dataset:** Karkkainen, K., & Joo, J. (2021). *FairFace: Face Attribute Dataset with Balanced Race, Gender, and Age for Bias Measurement and Mitigation.* IEEE/CVF Winter Conference on Applications of Computer Vision (WACV).
* **PyTorch & Torchvision:** Deep learning framework & torchvision residual backbones.
* **Streamlit & Streamlit-WebRTC:** Web dashboard framework & real-time WebRTC streaming wrappers.
 
