# Import necessary library

# For managing audio file
import librosa
from pydub import AudioSegment
from pydub.silence import split_on_silence

# Importing Pytorch
import torch
import numpy as np

# Importing Wav2Vec
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

import io
import scipy.io.wavfile


class NepaliASR:
    def __init__(self):
        # self.processor = Wav2Vec2Processor.from_pretrained("./content/audio-bee-trained-nepali")
        # self.model = Wav2Vec2ForCTC.from_pretrained("./content/audio-bee-trained-nepali")
        # self.processor_default = Wav2Vec2Processor.from_pretrained("gagan3012/wav2vec2-xlsr-nepali")
        # self.model_default = Wav2Vec2ForCTC.from_pretrained("gagan3012/wav2vec2-xlsr-nepali").to("cuda")

        self.processor_default = Wav2Vec2Processor.from_pretrained("./models/wav2vec2-xlsr-nepali")
        self.model_default = Wav2Vec2ForCTC.from_pretrained("./models/wav2vec2-xlsr-nepali").to("cuda")

    def speech_recognition(self, filename):
        audio, rate = librosa.load(filename, sr=16000)
        inputs = self.processor(audio, sampling_rate=16_000, return_tensors="pt", padding=True)

        with torch.no_grad():
            logits = self.model(inputs.input_values.to("cuda"), attention_mask=inputs.attention_mask.to("cuda")).logits

        prediction = torch.argmax(logits, dim=-1)

        transcription = self.processor.batch_decode(prediction)[0]

        return (transcription)

    def speech_recognition_default(self, filename):
        audio, rate = librosa.load(filename, sr=16000)

        audio_segment = file_to_audiosegment(audio, rate)

        audio_chunks = split_on_silence(audio_segment, min_silence_len=750, silence_thresh=-40)
        transcriptions = []
        for audio_chunk in audio_chunks:
            transcriptions.append(self.transcribe_audio(audiosegment_to_librosawav(audio_chunk)))
        transcription = " ".join(transcriptions)
        return transcription

    def speech_recognition_file(self, audio, sample_rate=16000):

        audio_segment = file_to_audiosegment(audio, sample_rate)

        audio_chunks = split_on_silence(audio_segment, min_silence_len=750, silence_thresh=-40)
        transcriptions = []
        for audio_chunk in audio_chunks:
            transcriptions.append(self.transcribe_audio(audiosegment_to_librosawav(audio_chunk)))

        transcription = " ".join(transcriptions)
        return transcription

    def transcribe_audio(self, audio_segment):
        inputs = self.processor_default(audio_segment, sampling_rate=16_000,
                                        return_tensors="pt", padding=True).to("cuda")

        with torch.no_grad():
            input = inputs.input_values.to("cuda")
            input_am = inputs.attention_mask.to("cuda")
            logits = self.model_default(input, attention_mask=input_am).logits

        prediction = torch.argmax(logits, dim=-1)
        return self.processor_default.batch_decode(prediction)[0]


def audiosegment_to_librosawav(audiosegment):
    audiosegment = audiosegment.set_frame_rate(16000)
    channel_sounds = audiosegment.split_to_mono()
    samples = [s.get_array_of_samples() for s in channel_sounds]

    fp_arr = np.array(samples).T.astype(np.float32)
    fp_arr /= np.iinfo(samples[0].typecode).max
    fp_arr = fp_arr.reshape(-1)

    return fp_arr


def file_to_audiosegment(audio, sample_rate):
    wav_io = io.BytesIO()
    scipy.io.wavfile.write(wav_io, sample_rate, audio)
    wav_io.seek(0)
    audio_segment = AudioSegment.from_wav(wav_io)
    return audio_segment



