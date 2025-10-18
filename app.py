from flask import Flask, request, jsonify
import librosa
import numpy as np

app = Flask(__name__)

def predict_emotion_from_audio(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr).T, axis=0)
    energy = np.mean(np.abs(y))
    if energy < 0.1:
        return "buồn bã"
    elif energy < 0.3:
        return "lo âu"
    else:
        return "tức giận"

@app.route("/detect_emotion", methods=["POST"])
def detect_emotion():
    audio = request.files["audio"]
    audio.save("temp.wav")
    emotion = predict_emotion_from_audio("temp.wav")
    return jsonify({"emotion": emotion})

