# Voice Emotion Detection Web App

This project is a **voice emotion detection web application** built with **Flask**, **TensorFlow/Keras**, and **librosa**. It lets you:

- Record your voice directly from the browser.
- Send the audio to a Python backend.
- Run a **4-class CNN model** that predicts one of:
  - `angry`
  - `sad`
  - `happy`
  - `natural` (neutral)

The model is trained on a local dataset of WAV files stored in the `dataset/` folder.

---

## 1. Project structure

Key files and folders:

- `app.py`  
  Flask backend that:
  - Serves the main HTML page (`/`).
  - Receives recorded audio at `/analyze`.
  - Extracts MFCC features and runs the trained CNN model.

- `train_model.py`  
  Training script that:
  - Loads WAV files from the dataset.
  - Extracts MFCC features with `librosa`.
  - Trains a 4-class CNN.
  - Saves the model to `model/Emotion_Voice_Detection_Model.h5`.

- `templates/index.html`  
  Frontend HTML page with a modern UI for:
  - Recording voice.
  - Showing status messages.
  - Playing back the recording.
  - Displaying the predicted emotion.

- `static/js/record.js`  
  Frontend JavaScript that:
  - Uses the browser microphone (`MediaRecorder`) to capture audio.
  - Sends the recording to `/analyze` via `fetch`.
  - Updates the UI with the predicted emotion.

- `dataset/`  
  Contains the training data, organized as:

  ```text
  dataset/
    Angry/   *.wav
    Sad/     *.wav
    Happy/   *.wav
    Neutral/ *.wav
  ```

- `model/Emotion_Voice_Detection_Model.h5`  
  Trained Keras model (created by `train_model.py`).

---

## 2. Requirements

Python packages are listed in `requirements.txt`:

```text
flask
librosa
soundfile
numpy
tensorflow
```

Install them (preferably in a virtual environment):

```bash
pip install -r requirements.txt
```

### ffmpeg (required by librosa)

`librosa` and `soundfile` may require **ffmpeg** to read audio formats such as WebM. On Windows:

1. Download a static build from a trusted source (e.g. `https://www.gyan.dev/ffmpeg/builds/`).
2. Extract it, e.g. to `C:\ffmpeg`.
3. Add `C:\ffmpeg\bin` to your `PATH` environment variable.

After that, restart the terminal and verify:

```bash
ffmpeg -version
```

---

## 3. Dataset format

The training script expects WAV files organized as:

```text
dataset/
  Angry/   *.wav   -> label 0 ("angry")
  Sad/     *.wav   -> label 1 ("sad")
  Happy/   *.wav   -> label 2 ("happy")
  Neutral/ *.wav   -> label 3 ("natural")
```

Each WAV file should be a single speech sample expressing the corresponding emotion.

You can add or remove files in these folders; the script will automatically load all `*.wav` files in each class folder.

---

## 4. How the model works

- **Feature extraction**
  - Audio is loaded with `librosa.load(..., sr=22050, mono=True)`.
  - MFCCs are computed with `n_mfcc = 40`.
  - Time axis is padded or truncated to a fixed length (`MAX_PAD_LEN = 174`).
  - Final input shape for the CNN is `(40, 174, 1)`.

- **Model architecture** (simplified)
  - 2 convolution + max-pooling blocks.
  - Dense layer with dropout.
  - Output layer with 4 units and softmax activation.

- **Labels**

  ```python
  EMOTION_LABELS = [
      "angry",   # 0
      "sad",     # 1
      "happy",   # 2
      "natural", # 3
  ]
  ```

---

## 5. Training the model

Run this once (or whenever you change the dataset) to train and save the model:

```bash
python train_model.py
```

What it does:

- Reads all WAV files from `dataset/Angry`, `dataset/Sad`, `dataset/Happy`, `dataset/Neutral`.
- Extracts MFCC features for each file.
- Trains the CNN for several epochs.
- Saves the model to `model/Emotion_Voice_Detection_Model.h5`.

If training runs successfully you will see logs about dataset size, epochs, and finally:

```text
Saving model to .../model/Emotion_Voice_Detection_Model.h5
Done.
```

You only need to retrain if you change the dataset or want to improve the model.

---

## 6. Running the web app

1. Make sure dependencies are installed and the model file exists (either trained by you or provided).
2. From the project root (`04-WebApp`), start the Flask server:

   ```bash
   python app.py
   ```

3. You should see output similar to:

   ```text
   Starting Flask app...
   MODEL_PATH: E:\\projects\\04-WebApp\\model\\Emotion_Voice_Detection_Model.h5
   * Serving Flask app 'app'
   * Debug mode: on
   ```

4. Open your browser and go to:

   ```text
   http://127.0.0.1:5000/
   ```

5. In the UI:
   - Allow microphone access when prompted.
   - Click **Record**, speak for a few seconds.
   - Click **Stop** to send the recording to the backend.
   - The page will display the predicted emotion (`angry`, `sad`, `happy`, or `natural`).

---

## 7. API endpoints

- `GET /`
  - Returns the main HTML page (`index.html`).

- `POST /analyze`
  - Expects a form-data field named `audio` containing the recorded audio blob.
  - Runs feature extraction and model prediction.
  - Returns JSON:

    ```json
    {
      "success": true,
      "emotion": "sad",
      "class_index": 1
    }
    ```

If there is an error (missing file, prediction error, etc.), the response will be:

```json
{
  "success": false,
  "error": "...details..."
}
```

---

## 8. Notes and tips

- Predictions depend strongly on the **quality and balance** of your dataset.
- If the model tends to output one emotion too often, consider:
  - Adding more samples for the under-represented classes.
  - Re-training the model with more epochs or different hyperparameters.
- You can customize the UI by editing `templates/index.html` and `static/js/record.js` without touching the backend.

---

## 9. License

This project is for educational and experimental purposes. Adjust licensing information here according to your needs.

