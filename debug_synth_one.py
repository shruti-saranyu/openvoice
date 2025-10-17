from faster_whisper import WhisperModel

audio_path = "output.wav"   # path to your file
model_size = "medium"       # "small" if you're low on RAM/CPU, "large" if you want best quality
device = "cuda"             # or "cpu"

model = WhisperModel(model_size, device=device, compute_type="float16" if device=="cuda" else "int8")
segments, info = model.transcribe(audio_path, beam_size=5)  # beam_size improves accuracy
print("Detected language code:", info.language)             # e.g., 'kn', 'en', 'hi', 'te' etc.
print("Language probability (approx):", getattr(info, "language_probability", "N/A"))

full_text = " ".join([seg.text for seg in segments]).strip()
print("\n--- TRANSCRIPTION ---\n", full_text)
print("\n--- SEGMENTS (first 5) ---")
for s in segments[:5]:
    print(f"{s.start:.2f}s -> {s.end:.2f}s : {s.text}")
