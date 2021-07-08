
from nepali_asr import NepaliASR


engine = NepaliASR()

while True:
    filename = input("Enter file name: ")
    if filename == "exit":
        break
    print(engine.speech_recognition_default(filename))
