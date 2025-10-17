from transformers import AutoModel
import numpy as np
import soundfile as sf

# Load IndicF5 from Hugging Face
repo_id = "ai4bharat/IndicF5"
model = AutoModel.from_pretrained(repo_id, trust_remote_code=True)

# Generate speech
audio = model(
    "गैस्ट्रिक की समस्या होना हर परिवार में एक या दो लोगों को अपने जीवनकाल में एक-दो बार तो आती ही है, यह बहुत सामान्य है।",
    ref_audio_path="my_voices/speaker_1/001_0000000_0007000.wav",
    ref_text="ಕಾಸ್ಟ್ರಿಕ್​ ಸಮಸೇನು ಪ್ರತ್ತಿ ಸಮಸೇನಲ್ಲಿ ಒಬ್ಬುರು ಅತ್ವಾ ಇಬ್ಬುರು ಅವರ್​ ಲೈಫ್ಟೆಮ್ನಲ್ಲಿ ಒಂದೆ ರಡಿಸಲ್ಲಿ ಆದರು ಬಂದಿರುವತ್ತದು ಸರ್ವೇಸಲ್ಲಿ."
)

# Normalize and save output
if audio.dtype == np.int16:
    audio = audio.astype(np.float32) / 32768.0
sf.write("namaste.wav", np.array(audio, dtype=np.float32), samplerate=24000)
print("Audio saved succesfully.")
