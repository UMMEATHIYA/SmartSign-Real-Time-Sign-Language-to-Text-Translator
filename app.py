import streamlit as st
import numpy as np
import cv2
import mediapipe as mp
from collections import deque
from tensorflow.keras.models import load_model
import time
import os

# Load model and label map
model = load_model('model/sign_lstm.h5')

# Invert label_map from utils.py
label_map = {}
for idx, label in enumerate(sorted(os.listdir('data/asl_alphabet_train'))):
    label_map[idx] = label

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Streamlit UI
st.title("🤟 SmartSign: Real-Time Sign Language Translator")
st.markdown("Use webcam to predict ASL signs live using MediaPipe + LSTM")
run = st.checkbox('Start Webcam')
FRAME_WINDOW = st.image([])

sequence = deque(maxlen=30)

cap = cv2.VideoCapture(0)
predicted_label = ''

while run:
    ret, frame = cap.read()
    if not ret:
        st.warning("Failed to grab frame.")
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        keypoints = []
        for lm in hand_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z])
        sequence.append(keypoints)

        # Draw hand landmarks
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        if len(sequence) == 30:
            input_seq = np.expand_dims(sequence, axis=0)  # shape: (1, 30, 63)
            prediction = model.predict(input_seq)
            class_id = np.argmax(prediction)
            predicted_label = label_map[class_id]

    # Display prediction
    cv2.putText(frame, f"Prediction: {predicted_label}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

cap.release()
