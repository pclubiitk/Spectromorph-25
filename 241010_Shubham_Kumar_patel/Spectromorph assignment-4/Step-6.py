import whisper
import torch
model = whisper.load_model("small")  # Or try "small", "medium", "large" if you want more accuracy
result = model.transcribe(r"C:\Users\DELL\Downloads\final_noise_audio.wav",word_timestamps=False)
print("🔊 Transcription:", result["text"])
print("Model device:", next(model.parameters()).device)