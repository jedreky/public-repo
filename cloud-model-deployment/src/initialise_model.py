import whisper

from src.utils import device

print("Initialising Whisper models")

models = ["base", "medium"]

for model in models:
    print(f"Model: {model}")
    model = whisper.load_model(name=model, device=device)
    result = model.transcribe("speech_sample.mp3")
    print("Output:" + result["text"])
