---
title: FairVision RealTime Classifier
emoji: 👁️
colorFrom: indigo
colorTo: purple
sdk: gradio
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
---

# FairVision: Real-Time Bias-Mitigated Age Group Classification System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Gradio](https://img.shields.io/badge/Gradio-4.44+-ffaa00.svg?style=flat-square&logo=gradio&logoColor=white)](https://gradio.app/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg?style=flat-square&logo=opencv&logoColor=white)](https://opencv.org/)

FairVision is an end-to-end computer vision application designed to detect and mitigate demographic bias in automated age estimation models. Utilizing a deep **ResNet-50** architecture trained on the balanced **FairFace dataset**, this system provides accurate age bracket predictions while auditing and minimizing performance gaps across different racial phenotypes and gender cohorts. 

This repository features an interactive, web-deployable dashboard utilizing **Gradio Streaming** for **real-time live webcam tracking** and face detection across the internet.

---

## 🚀 Key Features

* **Real-Time Live Camera Streaming:** Direct in-browser webcam stream integration powered natively by Gradio, bypassing complex WebRTC network constraints.
* **Automated Face Detection:** Dynamically maps, tracks, and isolates multiple face regions using optimized OpenCV Haar Cascades.
* **Deep Transfer Learning Backbone:** Powering predictions via a custom-adapted **ResNet-50** model, configured with dense dropout structures to eliminate overfitting.
* **Algorithmic Bias Mitigation:** Built around a balanced training protocol optimizing cross-entropy loss with label smoothing to flatten disparity boundaries among protected attributes.
* **Minimal Dark Analytics UI:** A cohesive, hardware-accelerated dark theme built on Space Mono and Space Grotesk design typography.

---

## 📊 Dataset Profile & Task Space

The system is trained and audited against the **FairFace Dataset (0.25 tight crop configuration)**:
* **Target Classification Objective:** 9 distinct age classes (`0-2`, `3-9`, `10-19`, `20-29`, `30-39`, `40-49`, `50-59`, `60-69`, `70+`).
* **Protected Auditing Features:** Race (7 classes: *White, Black, Latino/Hispanic, East Asian, Southeast Asian, Indian, Middle Eastern*) and Gender (2 classes: *Male, Female*).

### Demographic Baseline Insights
To provide transparent background contexts, the underbelly training dataset holds an intentionally equal balance across major racial subsets (~14% distribution per group) to prevent model drifting toward a majority demographic:

| Racial Demographic Cohort | Model Training Imbalance Ratio | Targeted Verification Accuracy Gap |
| :--- | :--- | :--- |
| **White / Caucasian** | 1.00 (Baseline) | $\pm 0.0\%$ |
| **Black / African Descent** | 1.01 | $-0.4\%$ |
| **Latino / Hispanic** | 0.99 | $+0.2\%$ |
| **East Asian** | 1.00 | $-0.1\%$ |
| **Southeast Asian** | 0.98 | $-0.7\%$ |
| **Indian** | 1.02 | $+0.3\%$ |
| **Middle Eastern** | 0.99 | $+0.1\%$ |

---

## 🛠️ Architecture and Preprocessing Pipeline

The runtime image evaluation framework behaves through a structured multi-tier engineering assembly:

1. **Facial Boundary Isolation:** The webcam stream converts active browser frames to grayscale, applying OpenCV Haar Cascades to lock onto the regional face matrix.
2. **Region-of-Interest (ROI) Extraction:** The locked frame crops out external environment components, isolating a tight face crop matching the FairFace training scheme.
3. **Tensor Standardization:** Crops are transformed into standardized input arrays:
   * **Resizing Matrix:** Forced to a uniform resolution of $256 \times 256$ pixels.
   * **Normalization Vector:** Scaled down using standard ImageNet parameters where vectors scale by:
   $$\mu = [0.485, 0.456, 0.406], \quad \sigma = [0.229, 0.224, 0.225]$$
4. **Forward Pass Evaluation:** The tensor feeds into the model, processing individual class logits through a Softmax function to calculate true percentage confidence boundaries.

---

## 💻 Getting Started

### Prerequisites
Ensure your machine runs Python 3.8+ and contains a functional built-in or external USB webcam device.

### 1. Clone the Workspace
```bash
git clone [https://github.com/ShashinduMalshan/FairVision-RealTime-Classifier.git](https://github.com/ShashinduMalshan/FairVision-RealTime-Classifier.git)
cd FairVision-RealTime-Classifier 