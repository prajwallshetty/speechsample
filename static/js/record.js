let mediaRecorder = null;
let chunks = [];

const recordBtn = document.getElementById("recordBtn");
const stopBtn = document.getElementById("stopBtn");
const statusEl = document.getElementById("status");
const playbackEl = document.getElementById("playback");
const emotionEl = document.getElementById("emotion");

async function startRecording() {
  emotionEl.textContent = "";
  playbackEl.style.display = "none";
  playbackEl.src = "";

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

    mediaRecorder = new MediaRecorder(stream);
    chunks = [];

    mediaRecorder.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) {
        chunks.push(e.data);
      }
    };

    mediaRecorder.onstop = async () => {
      const blob = new Blob(chunks, { type: "audio/webm" });
      const url = URL.createObjectURL(blob);
      playbackEl.src = url;
      playbackEl.style.display = "block";

      statusEl.textContent = "Analyzing emotion...";

      const formData = new FormData();
      formData.append("audio", blob, "recording.webm");

      try {
        const response = await fetch("/analyze", {
          method: "POST",
          body: formData,
        });

        const data = await response.json();

        if (data.success) {
          statusEl.textContent = "Analysis complete.";
          emotionEl.innerHTML = `Detected emotion: <span>${data.emotion}</span>`;
        } else {
          statusEl.textContent = "Error: " + (data.error || "Unknown error");
        }
      } catch (err) {
        console.error(err);
        statusEl.textContent = "Network error while analyzing audio.";
      }
    };

    mediaRecorder.start();
    statusEl.textContent = "Recording... Speak now.";
    recordBtn.disabled = true;
    stopBtn.disabled = false;
  } catch (err) {
    console.error(err);
    statusEl.textContent = "Could not access microphone. Please allow microphone permissions.";
  }
}

function stopRecording() {
  if (mediaRecorder && mediaRecorder.state === "recording") {
    mediaRecorder.stop();
    recordBtn.disabled = false;
    stopBtn.disabled = true;
    statusEl.textContent = "Processing recording...";
  }
}

recordBtn.addEventListener("click", startRecording);
stopBtn.addEventListener("click", stopRecording);

