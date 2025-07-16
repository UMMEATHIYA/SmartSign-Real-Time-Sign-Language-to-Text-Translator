# 🤟 SmartSign – Real-Time Sign Language to Text Translator

SmartSign is a real-time ASL (American Sign Language) recognition system that detects hand gestures using MediaPipe, classifies them using an LSTM model, and translates them into text using a Streamlit-based UI.

<img src="https://img.shields.io/badge/Python-3.11-blue?style=flat-square"/> <img src="https://img.shields.io/badge/Framework-Streamlit-orange"/> <img src="https://img.shields.io/badge/Model-LSTM-green"/>

---

## 🚀 Features
- Real-time webcam input for hand gesture detection
- MediaPipe-powered keypoint extraction
- LSTM model trained on synthetic time-series data
- Live Streamlit app with smooth prediction display
- Easy to extend to words/sentences or voice output

---

## 📁 Project Structure
```
SmartSign/
├── app.py # Streamlit real-time app
├── collect_data.py # (Optional) live keypoint capture
├── convert_images_to_sequences.py # Convert images → sequences
├── train_model.py # LSTM training script
├── utils.py # Helper functions to load and preprocess data
├── data/
│ ├── asl_alphabet_train/ # ASL image folders (A-Z)
│ ├── sequences/ # Generated CSVs (30×63 keypoints)
├── model/
│ └── sign_lstm.h5 # Trained model
├── run.sh # Bash script to launch app
└── README.md
```


---

## ⚙️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/UMMEATHIYA/SmartSign-Real-Time-Sign-Language-to-Text-Translator.git
cd SmartSign-Real-Time-Sign-Language-to-Text-Translator
```
### 2. Install Dependencies
```
pip install -r requirements.txt
```
<details> <summary>🔧 Or manually install:</summary></details>

```
pip install streamlit opencv-python mediapipe tensorflow numpy pandas
```

### 🧠 Train the Model (Optional)
If you want to retrain:
```
python convert_images_to_sequences.py
python train_model.py
```

### 🎯 Run the App
```
python -m streamlit run app.py
```
---
###✅ You’ll See Two URLs in the Terminal:
```
  Local URL: http://localhost:8501
  Network URL: http://10.0.0.50:8501
```
### 🧪 Demo Output
<img width="1490" height="663" alt="image" src="https://github.com/user-attachments/assets/3a1dbdef-ec95-4221-807b-10ffdf9ef376" />

---
<img width="1087" height="872" alt="image" src="https://github.com/user-attachments/assets/708f4e7e-5aca-467e-89d8-73c5e302b36c" />


### 💬 Contact
Made with ❤️ by Umme Athiya
📧 uathiya4@gmail.com
🌐 Portfolio | GitHub

---

```bash
#!/bin/bash

echo "✅ Activating SmartSign..."
echo "🔁 Starting Streamlit app..."

python -m streamlit run app.py

