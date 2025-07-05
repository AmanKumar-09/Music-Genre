from flask import Flask, request, jsonify
import librosa
import numpy as np
import joblib
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model = joblib.load("../ML_Model/genre_classifier.pkl")  # Load your trained model

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    filepath = "temp.wav"
    file.save(filepath)

    try:
        y, sr = librosa.load(filepath, duration=30)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_mean = np.mean(mfcc.T, axis=0)

        prediction = model.predict([mfcc_mean])[0]
        os.remove(filepath)

        return jsonify({"genre": prediction})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
