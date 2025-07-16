import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

# PATH SETUP
DATASET_PATH = "data/asl_alphabet_train"
OUTPUT_PATH = "data/sequences"
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

# Get label folders (A-Z)
labels = sorted([label for label in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, label))])

for label in labels:
    label_folder = os.path.join(DATASET_PATH, label)
    images = [img for img in os.listdir(label_folder) if img.endswith(".jpg")]

    print(f"Processing label '{label}'...")

    # Process in chunks of 30
    for i in range(0, len(images) - 30, 30):
        sequence = []
        for j in range(i, i + 30):
            img_path = os.path.join(label_folder, images[j])
            image = cv2.imread(img_path)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            if result.multi_hand_landmarks:
                keypoints = []
                for lm in result.multi_hand_landmarks[0].landmark:
                    keypoints.extend([lm.x, lm.y, lm.z])
                sequence.append(keypoints)
            else:
                break  # Skip this sequence if one frame fails

        # Save only if full sequence is valid
        if len(sequence) == 30:
            df = pd.DataFrame(sequence)
            filename = f"{label}_{i}.csv"
            df.to_csv(os.path.join(OUTPUT_PATH, filename), index=False)
            print(f"Saved: {filename}")
        else:
            print(f"Skipped incomplete sequence: {label}_{i}")
