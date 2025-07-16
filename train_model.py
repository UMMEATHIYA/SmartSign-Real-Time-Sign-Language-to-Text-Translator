from utils import load_data
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import os

# Load the data
X, y, label_map = load_data()
y_cat = to_categorical(y)

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y_cat, test_size=0.2, random_state=42)

# Build LSTM model
model = Sequential([
    LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 63)),
    LSTM(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(len(label_map), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50)

# Save the model
os.makedirs("model", exist_ok=True)
model.save("model/sign_lstm.h5")
print("Model saved at model/sign_lstm.h5")
