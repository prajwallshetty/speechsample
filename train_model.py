import os
from glob import glob

import numpy as np
import librosa
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical


CLASS_NAMES = ["angry", "sad", "happy", "natural"]
NUM_CLASSES = len(CLASS_NAMES)
N_MFCC = 40
MAX_PAD_LEN = 174


def build_model(input_shape=(N_MFCC, MAX_PAD_LEN, 1), num_classes=NUM_CLASSES) -> Model:
    inp = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation="relu", padding="same")(inp)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    out = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def extract_features(file_path: str) -> np.ndarray:
    """Extract MFCC features (same style as main.py)."""

    y, sr = librosa.load(file_path, sr=22050, mono=True)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)

    if mfcc.shape[1] < MAX_PAD_LEN:
        pad_width = MAX_PAD_LEN - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode="constant")
    else:
        mfcc = mfcc[:, :MAX_PAD_LEN]

    return mfcc


def load_dataset(dataset_root: str) -> tuple[np.ndarray, np.ndarray]:
    X = []
    y = []

    for idx, class_name in enumerate(CLASS_NAMES):
        if class_name == "angry":
            folder = "Angry"
        elif class_name == "sad":
            folder = "Sad"
        elif class_name == "happy":
            folder = "Happy"
        elif class_name == "natural":
            folder = "Neutral"
        else:
            continue

        class_dir = os.path.join(dataset_root, folder)
        pattern = os.path.join(class_dir, "*.wav")
        files = glob(pattern)
        print(f"Loading {len(files)} files for class '{class_name}' from {class_dir}")

        for f in files:
            try:
                feats = extract_features(f)
                X.append(feats)
                y.append(idx)
            except Exception as e:
                print(f"Failed to process {f}: {e}")

    X = np.array(X, dtype="float32")
    X = X[..., np.newaxis]
    y = np.array(y)
    y_cat = to_categorical(y, num_classes=NUM_CLASSES)
    return X, y_cat


def main():
    base_dir = os.path.dirname(__file__)
    dataset_root = os.path.join(base_dir, "dataset")

    print(f"Loading dataset from {dataset_root} ...")
    X, y = load_dataset(dataset_root)
    print(f"Dataset loaded: X shape = {X.shape}, y shape = {y.shape}")

    print("Building model...")
    model = build_model()

    print("Training model on real data...")
    model.fit(X, y, epochs=30, batch_size=16, validation_split=0.15)

    model_dir = os.path.join(base_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "Emotion_Voice_Detection_Model.h5")

    print(f"Saving model to {model_path} ...")
    model.save(model_path)
    print("Done.")


if __name__ == "__main__":
    main()

