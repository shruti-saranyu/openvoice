import numpy as np
import soundfile as sf
from transformers import AutoModel

repo_id = "ai4bharat/IndicF5"
model = AutoModel.from_pretrained(repo_id, trust_remote_code=True)

# Example inputs
text = "ప్రతి కుటుంబంలో ఒకరు లేదా ఇద్దరు వ్యక్తులకు వారి జీవితకాలంలో కనీసం ఒకటి లేదా రెండుసార్లు గ్యాస్ట్రిక్ సమస్యలు రావడం చాలా సాధారణం."
ref_audio_path = "my_voices/speaker_1/001_0000000_0007000.wav"
ref_text = "ಕಾಸ್ಟ್ರಿಕ್​ ಸಮಸೇನು ಪ್ರತ್ತಿ ಸಮಸೇನಲ್ಲಿ ಒಬ್ಬುರು ಅತ್ವಾ ಇಬ್ಬುರು ಅವರ್​ ಲೈಫ್ಟೆಮ್ನಲ್ಲಿ ಒಂದೆ ರಡಿಸಲ್ಲಿ ಆದರು ಬಂದಿರುವತ್ತದು ಸರ್ವೇಸಲ್ಲಿ."

audio = model(text, ref_audio_path=ref_audio_path, ref_text=ref_text)

# If output is int16, convert to float
if audio.dtype == np.int16:
    audio = audio.astype(np.float32) / 32768.0

sf.write("output.wav", np.array(audio, dtype=np.float32), samplerate=24000)
print("Saved output.wav")
