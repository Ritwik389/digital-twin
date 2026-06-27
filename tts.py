"""
tts.py
Lightweight Voice cloning for the Digital Twin using Kyutai Pocket TTS.
Runs blazingly fast on CPUs for easy hosting.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import torch

# Ensure environment variables (like HF_TOKEN) are loaded
load_dotenv(".env.local")
torch.set_num_threads(os.cpu_count())

REFERENCE_WAV = "jensen_ref.wav"
OUTPUT_WAV = "response.wav"

_model = None
_voice_state = None
_load_error = None

def _load_model():
    global _model, _voice_state, _load_error

    if _model is not None or _load_error is not None:
        return

    if not Path(REFERENCE_WAV).exists():
        _load_error = f"{REFERENCE_WAV} not found."
        return

    try:
        from pocket_tts import TTSModel
        
        print("[tts] Loading Pocket TTS (CPU optimized)...")
        # Hugging Face will automatically look for os.environ["HF_TOKEN"] to bypass the gate
        _model = TTSModel.load_model()
        
        print(f"[tts] Generating voice state from {REFERENCE_WAV}...")
        _voice_state = _model.get_state_for_audio_prompt(REFERENCE_WAV)
        
        print("[tts] Pocket TTS ready")

    except Exception as e:
        _load_error = f"Pocket TTS failed to load: {e}"
        print(f"[tts] {_load_error}")

def is_available() -> bool:
    return Path(REFERENCE_WAV).exists()

def get_error() -> str | None:
    return _load_error

def speak(text: str, output_path: str = OUTPUT_WAV) -> str | None:
    _load_model()
    if _model is None or _voice_state is None:
        return None
    if not text.strip():
        return None

    global _load_error
    try:
        import scipy.io.wavfile
        
        # Generate the audio tensor
        audio = _model.generate_audio(_voice_state, text)
        
        # --- VOLUME FIX: Peak Normalization ---
        # Find the loudest peak in the generated audio
        max_val = torch.max(torch.abs(audio))
        if max_val > 0:
            # Boost the entire audio file so the peak hits 95% max volume
            audio = (audio / max_val) * 0.95 
            
        scipy.io.wavfile.write(output_path, _model.sample_rate, audio.numpy())
        return output_path
        
    except Exception as e:
        _load_error = f"speak failed: {e}"
        print(f"[tts] {_load_error}")
        return None
    
if __name__ == "__main__":
    if not is_available():
        print(f"[tts] {REFERENCE_WAV} not found")
    else:
        out = speak("This is a test of the Pocket TTS CPU voice cloning system.")
        print(f"[tts] saved to {out}" if out else f"[tts] Error: {get_error()}")