from flask import Flask, render_template, request, jsonify
import os
from datetime import datetime

import librosa
import numpy as np
from tensorflow.keras.models import load_model


app = Flask(__name__)


EMOTION_LABELS = [
    "angry",
    "sad",
    "happy",
    "natural",
]


N_MFCC = 40
MAX_PAD_LEN = 174


MODEL_PATH = os.path.join(app.root_path, "model", "Emotion_Voice_Detection_Model.h5")
MODEL = load_model(MODEL_PATH)


def extract_features(file_path: str) -> np.ndarray:
    """Extract MFCC features consistent with train_model.py."""

    y, sr = librosa.load(file_path, sr=22050, mono=True)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)

    if mfcc.shape[1] < MAX_PAD_LEN:
        pad_width = MAX_PAD_LEN - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode="constant")
    else:
        mfcc = mfcc[:, :MAX_PAD_LEN]

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
        features = extract_features(save_path)
        features = features[np.newaxis, ..., np.newaxis]

        probs = MODEL.predict(features)
        pred_index = int(np.argmax(probs, axis=1)[0])
        emotion = EMOTION_LABELS[pred_index] if 0 <= pred_index < len(EMOTION_LABELS) else "unknown"
    except Exception as e:
        return jsonify({"success": False, "error": f"Prediction error: {str(e)}"}), 500

    return jsonify({"success": True, "emotion": emotion, "class_index": pred_index})


if __name__ == "__main__":
    print("Starting Flask app...")
    print("MODEL_PATH:", MODEL_PATH)
    app.run(host="0.0.0.0", port=5000, debug=True)

