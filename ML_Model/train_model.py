import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

DATA_PATH = r"C:\Users\Administrator\Downloads\music_genre_classifier\data\genre"

genres = os.listdir(DATA_PATH)
features = []
labels = []

print("Extracting features...")

for genre in genres:
    genre_path = os.path.join(DATA_PATH, genre)
    for file in tqdm(os.listdir(genre_path)):
        try:
            file_path = os.path.join(genre_path, file)
            audio, sr = librosa.load(file_path, duration=30)
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
            mfcc_mean = np.mean(mfcc.T, axis=0)
            features.append(mfcc_mean)
            labels.append(genre)
        except Exception as e:
            print(f"Error: {e}")

# Prepare dataset
X = np.array(features)
y = np.array(labels)

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save model
joblib.dump(model, "genre_classifier.pkl")
print("Model saved as genre_classifier.pkl")
