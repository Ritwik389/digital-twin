"""
tts.py
Voice cloning for the Jensen Huang Digital Twin, using Qwen3-TTS
(Qwen/Qwen3-TTS-12Hz-0.6B-Base by default).

How it works:
  - Clones Jensen's voice from the short reference clip `jensen_ref.wav`
    (produced by data_collector.py's extract_reference_clip()).
  - A transcript of that clip (`jensen_ref.txt`) gives the best cloning
    quality (in-context-learning / ICL mode). If it's missing, it is
    auto-transcribed once with Whisper (already used by
    data_collector.py) and cached to disk. If Whisper isn't available
    either, falls back to x-vector-only cloning, which needs no
    transcript but is slightly lower fidelity.
  - The voice-clone "prompt" (reference features) is built ONCE on
    first use and reused for every line Jensen speaks afterwards, per
    the Qwen3-TTS docs (avoids recomputing reference features per turn).

Heavy imports (torch, qwen_tts) only happen inside _load_model(), so
importing this module is cheap even if those packages aren't installed
yet — speak() will simply fail gracefully and report the error via
get_error().
"""

import os
import re
from pathlib import Path

REFERENCE_WAV = "jensen_ref.wav"
REFERENCE_TXT = "jensen_ref.txt"
OUTPUT_WAV = "response.wav"
MODEL_NAME = os.environ.get("QWEN_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-Base")
LANGUAGE = "English"
MAX_WORDS = 250  # truncate long answers to keep synthesis latency reasonable

_model = None        # Qwen3TTSModel, loaded lazily
_clone_prompt = None  # cached VoiceClonePromptItem list, built once
_load_error = None    # last load/runtime error, surfaced to the UI


def _get_ref_text() -> str | None:
    """Return the transcript for jensen_ref.wav, transcribing it once if needed."""
    txt_path = Path(REFERENCE_TXT)
    if txt_path.exists():
        text = txt_path.read_text(encoding="utf-8").strip()
        return text or None

    if not Path(REFERENCE_WAV).exists():
        return None

    try:
        import whisper
        print("[tts] transcribing jensen_ref.wav for the voice-clone prompt (one-time)...")
        model = whisper.load_model("base")
        result = model.transcribe(REFERENCE_WAV)
        text = result["text"].strip()
        if text:
            txt_path.write_text(text, encoding="utf-8")
            return text
    except Exception as e:
        print(f"[tts] could not transcribe reference clip: {e}")

    return None


def _load_model():
    global _model, _clone_prompt, _load_error

    if _model is not None or _load_error is not None:
        return

    if not Path(REFERENCE_WAV).exists():
        _load_error = f"{REFERENCE_WAV} not found - run data_collector.py to extract it."
        return

    try:
        import torch
        from qwen_tts import Qwen3TTSModel

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        load_kwargs = {"device_map": device, "dtype": dtype}
        if torch.cuda.is_available():
            try:
                import flash_attn  # noqa: F401
                load_kwargs["attn_implementation"] = "flash_attention_2"
            except ImportError:
                pass

        print(f"[tts] loading {MODEL_NAME} on {device}...")
        model = Qwen3TTSModel.from_pretrained(MODEL_NAME, **load_kwargs)

        ref_text = _get_ref_text()
        x_vector_only = ref_text is None
        if x_vector_only:
            print("[tts] no transcript for jensen_ref.wav - using x-vector-only cloning")

        clone_prompt = model.create_voice_clone_prompt(
            ref_audio=REFERENCE_WAV,
            ref_text=ref_text,
            x_vector_only_mode=x_vector_only,
        )

        _model = model
        _clone_prompt = clone_prompt
        print("[tts] Qwen3-TTS ready")

    except Exception as e:
        _load_error = f"Qwen3-TTS failed to load: {e}"
        print(f"[tts] {_load_error}")

def _clean_for_tts(text: str) -> str:
    """Remove markdown and special characters that can confuse TTS."""
    text = re.sub(r'\*{1,2}([^*]+)\*{1,2}', r'\1', text)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'[`~|<>{}[\]]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def is_available() -> bool:
    """Whether the voice toggle should be shown (reference clip present)."""
    return Path(REFERENCE_WAV).exists()


def get_error() -> str | None:
    """Return the last load/runtime error message, if any."""
    return _load_error


def speak(text: str, output_path: str = OUTPUT_WAV) -> str | None:
    """
    Synthesize `text` in Jensen's cloned voice.
    Returns the path to the generated WAV file, or None on failure
    (call get_error() for details).
    """
    _load_model()
    if _model is None:
        return None

    words = text.split()
    if len(words) > MAX_WORDS:
        text = " ".join(words[:MAX_WORDS]) + "..."

    text = _clean_for_tts(text)
    if not text.strip():
        return None

    global _load_error
    try:
        import soundfile as sf
        wavs, sr = _model.generate_voice_clone(
            text=text,
            language=LANGUAGE,
            voice_clone_prompt=_clone_prompt,
        )
        sf.write(output_path, wavs[0], sr)
        return output_path
    except Exception as e:
        _load_error = f"speak failed: {e}"
        print(f"[tts] {_load_error}")
        return None




if __name__ == "__main__":
    if not is_available():
        print(f"[tts] {REFERENCE_WAV} not found")
    else:
        out = speak(
            "The thing that's extraordinary about this moment is that AI is not just a product. "
            "It is a new way of doing computing."
        )
        print(f"[tts] saved to {out}" if out else f"[tts] no output generated: {get_error()}")
