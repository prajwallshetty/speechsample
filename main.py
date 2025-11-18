from flask import Flask, render_template, request, jsonify
import os
from datetime import datetime

import librosa
import numpy as np
from tensorflow.keras.models import load_model


app = Flask(__name__)


EMOTION_LABELS = [
    "female_angry",
    "female_calm",
    "female_fearful",
    "female_happy",
    "female_sad",
    "male_angry",
    "male_calm",
    "male_fearful",
    "male_happy",
    "male_sad",
]


MODEL_PATH = os.path.join(app.root_path, "model", "Emotion_Voice_Detection_Model.h5")
MODEL = load_model(MODEL_PATH)


def extract_features(file_path: str, max_pad_len: int = 174) -> np.ndarray:
    """Extract MFCC features similar to common SER setups.

    NOTE: This is a best-guess pipeline. If you trained with a different
    feature shape (e.g., different n_mfcc or time steps), we may need to
    adjust n_mfcc or max_pad_len.
    """

    # Load audio (mono) with a fixed sample rate for consistency
    y, sr = librosa.load(file_path, sr=22050, mono=True)

    # Compute MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

    # Pad or truncate to a fixed length on the time axis
    if mfcc.shape[1] < max_pad_len:
        pad_width = max_pad_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode="constant")
    else:
        mfcc = mfcc[:, :max_pad_len]

    return mfcc


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    if "audio" not in request.files:
        return jsonify({"success": False, "error": "No audio file provided"}), 400

    audio_file = request.files["audio"]
    if audio_file.filename == "":
        return jsonify({"success": False, "error": "Empty filename"}), 400

    uploads_dir = os.path.join(app.root_path, "uploads")
    os.makedirs(uploads_dir, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
    filename = f"recording_{timestamp}.webm"
    save_path = os.path.join(uploads_dir, filename)
    audio_file.save(save_path)

    try:
        # Extract features from recorded audio
        features = extract_features(save_path)
        # Reshape for model: (batch, features, time, channels)
        features = features[np.newaxis, ..., np.newaxis]

        # Predict probabilities and pick the highest
        probs = MODEL.predict(features)
        pred_index = int(np.argmax(probs, axis=1)[0])
        emotion = EMOTION_LABELS[pred_index] if 0 <= pred_index < len(EMOTION_LABELS) else "unknown"
    except Exception as e:
        # Fallback if anything goes wrong during prediction
        return jsonify({"success": False, "error": f"Prediction error: {str(e)}"}), 500

    return jsonify({"success": True, "emotion": emotion, "class_index": pred_index})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

