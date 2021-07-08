from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import os
import json
from urllib.request import urlretrieve
from dotenv import load_dotenv

from nepali_asr import NepaliASR

import librosa

# Load env variables
load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.getenv("UPLOAD_FOLDER")

engine = NepaliASR()


@app.route("/", methods=['GET'])
def home():
    return render_template("index.html")

@app.route("/speechrecognitionfile", methods=['POST'])
def speechrecognitionfile():
    if len(request.files) == 0:
        return "Please upload a file"
    file = request.files['file']
    filename = secure_filename(file.filename)
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(path)
    audio, rate = librosa.load(path, sr=16000)
    transcript = engine.speech_recognition_file(audio, rate)
    payload = {"transcript": transcript}
    return json.dumps(payload)


@app.route("/speechrecognitiondefault", methods=['POST'])
def speechrecognitiondefault():
    file = request.files['file']
    filename = secure_filename(file.filename)
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(path)
    transcript = engine.speech_recognition_default(path)
    payload = {"transcript": transcript}
    return json.dumps(payload)


@app.route("/speechrecognitionurl", methods=['POST'])
def speechrecognitionurl():
    test = request.json
    file = urlretrieve(test['url'])
    transcript = engine.speech_recognition_default(file[0])
    payload = {"transcript": transcript}
    return json.dumps(payload)


@app.route("/test", methods=['POST'])
def test():
    transcript = engine.speech_recognition_default("./content/test_audio/1.wav")
    return transcript


if __name__ == '__main__':
    app.run(port=os.getenv('PORT'))
