# FairVision AI

An AI-powered web application for age group classification using a deep learning model (ResNet-50) built with PyTorch and deployed using Streamlit.

---

## Live Demo

Try the application here:  
https://fairvision-ageclassifier-dynrx6uirepdtfcujj6mqm.streamlit.app/

---

## Project Overview

FairVision AI predicts the age group of a person from a face image using a trained deep learning model.  
The model is based on a modified ResNet-50 architecture and classifies images into 9 age categories.

---

## Age Categories

- 0–2  
- 3–9  
- 10–19  
- 20–29  
- 30–39  
- 40–49  
- 50–59  
- 60–69  
- 70+  

---

## Features

- Upload face images (JPG, PNG, JPEG)
- Real-time AI inference
- Age group prediction
- Confidence score display
- Probability distribution visualization
- Top-3 predictions

---

## Tech Stack

- Python  
- PyTorch  
- TorchVision  
- Streamlit  
- Pillow  
- gdown  

---

## Model Architecture

- Backbone: ResNet-50  
- Fully Connected Layer customized for 9 classes  
- Dropout (0.6) for regularization  
- Output: Softmax probabilities  

---

## How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/your-username/fairvision.git
cd fairvision
````

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the application

```bash
streamlit run app.py
```

---

## Model File Handling

* The model file (`FairVision.pt`) is automatically downloaded from Google Drive on first run.
* It is NOT included in the repository due to its large size.

---

## Project Structure

```md
FairVision_Web/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── FairVision.pt (auto-downloaded at runtime)
```

---

## Important Notes

* Ensure internet connection for first run (model download required)
* Do not commit `.pt` model files to GitHub
* Streamlit Cloud handles dependencies via `requirements.txt`

---

## Future Improvements

* Add face detection before prediction
* Integrate Grad-CAM explainability
* Improve dataset fairness and balance
* Deploy using Docker or HuggingFace Spaces

---

# FairVision-RealTime-Classifier
