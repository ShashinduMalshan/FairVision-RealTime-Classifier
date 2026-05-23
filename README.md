# FairVision: Real-Time Bias-Mitigated Age Group Classification System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-ff4b4b.svg?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg?style=flat-square&logo=opencv&logoColor=white)](https://opencv.org/)

FairVision is an end-to-end computer vision application designed to detect and mitigate demographic bias in automated age estimation models. Utilizing a deep **ResNet-50** architecture trained on the balanced **FairFace dataset**, this system provides accurate age bracket predictions while auditing and minimizing performance gaps across different racial phenotypes and gender cohorts. 

This repository expands upon the core research notebook by introducing a production-ready, interactive **Streamlit Dashboard** featuring **real-time live webcam tracking** and face detection.

---

## 🚀 Key Features

* **Dual-Inference Pipeline:**
  * **Static Upload Mode:** Processes standard image files (`.jpg`, `.png`, `.jpeg`), rendering detailed confidence metrics and probability distribution charts.
  * **Real-Time Live Camera Mode:** Direct webcam stream integration that maps and isolates faces dynamically using an automated Haar Cascade face detector.
* **Deep Transfer Learning Backbone:** Powering predictions via a custom-adapted **ResNet-50** model, configured with custom heavy dropout structures to restrict overfitting.
* **Algorithmic Bias Mitigation:** Built around a balanced training protocol optimizing cross-entropy loss with label smoothing to flatten disparity boundaries among protected attributes.
* **Dynamic Analytics UI:** Fully integrated visualization tracking that breaks down the model's top candidates and probability outputs using a clean, custom dark-themed dashboard layout.

---

## 📊 Dataset Profile & Task Space

The system is trained and audited against the **FairFace Dataset (0.25 tight crop configuration)**:
* **Target Classification Objective:** 9 distinct age classes (`0-2`, `3-9`, `10-19`, `20-29`, `30-39`, `40-49`, `50-59`, `60-69`, `70+`).
* **Protected Auditing Features:** Race (7 classes: *White, Black, Latino/Hispanic, East Asian, Southeast Asian, Indian, Middle Eastern*) and Gender (2 classes: *Male, Female*).

---

## 🛠️ Architecture and Preprocessing Pipeline

The runtime image evaluation framework behaves through a structured multi-tier engineering assembly:

1. **Facial Boundary Isolation:** The webcam stream converts active frames to grayscale, applying OpenCV Haar Cascades to lock onto the regional face matrix.
2. **Region-of-Interest (ROI) Extraction:** The locked frame crops out external neck and environmental background components, isolating a tight face crop matching the FairFace training scheme.
3. **Tensor Standardization:** Crops are transformed into standardized input arrays:
   * **Resizing Matrix:** Forced to a uniform resolution of $256 \times 256$ pixels.
   * **Normalization Vector:** Scaled down using standard ImageNet parameters (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).
4. **Forward Pass Evaluation:** The tensor feeds into the model, processing individual class logits through a Softmax function to calculate true percentage confidence boundaries.

---

## 💻 Getting Started

### Prerequisites
Ensure your machine runs Python 3.8+ and contains a functional built-in or external USB webcam device.

### 1. Clone the Workspace
```bash
git clone [https://github.com/ShashinduMalshan/FairVision-RealTime-Classifier.git](https://github.com/ShashinduMalshan/FairVision-RealTime-Classifier.git)
cd FairVision-RealTime-Classifier

```

### 2. Install Required Packages

```bash
pip install streamlit torch torchvision opencv-python pillow pandas numpy gdown

```

### 3. Run the Application

Start the Streamlit application framework locally:

```bash
streamlit run app.py

```

*Note: On your first startup, the system will automatically communicate with Google Drive storage using `gdown` to safely pull down your pre-trained weights (`FairVision.pt`) into your root directory.*

---

## ⚖️ Fairness Disclaimer & Ethical Use

FairVision is built as an educational and auditing framework to study the variance of computer vision across distinct human phenotypes. Models output purely statistical evaluations. Predictions are probabilistic and are **not suitable** for deployment in unmonitored surveillance networks, biometric high-stakes legal profiling, or any secure verification contexts where prediction errors could inflict demographic harm.
