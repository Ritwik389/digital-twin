"""Pocket TTS conditioned on Jensen's reference, with isolated audio files."""
import logging
import os
from pathlib import Path
import re
import threading
import time
import uuid

logger = logging.getLogger(__name__)
AUDIO_DIR = Path(os.getenv('AUDIO_DIR', 'audio_cache'))
LANGUAGE = os.getenv('POCKET_TTS_LANGUAGE', 'english')
REFERENCE_AUDIO = Path(os.getenv('POCKET_TTS_REFERENCE', 'jensen_ref.wav'))
_model = None
_voice_state = None
_load_error = None
_retry_after = 0.0
_lock = threading.RLock()


class VoiceSetupError(RuntimeError):
    pass


def is_loaded():
    # A partially loaded model cannot synthesize the cloned voice.
    return _model is not None and _voice_state is not None


def _retry_cloning_download():
    # Pocket catches every download exception and silently loads its stock-voice
    # weights. Retry the actual cloning download once so transient failures can
    # recover and a rejected token is distinguishable from a network outage.
    from pocket_tts.utils.config import CONFIGS_DIR, load_config
    from pocket_tts.utils.utils import download_if_necessary
    config = load_config(CONFIGS_DIR / f'{LANGUAGE}.yaml')
    try:
        download_if_necessary(config.weights_path)
    except Exception as exc:
        status = getattr(getattr(exc, 'response', None), 'status_code', None)
        logger.error('Cloning weight download failed: %s; HTTP status=%s', type(exc).__name__, status)
        if status in (401, 403):
            raise VoiceSetupError(f'Hugging Face rejected the cloning model download (HTTP {status}). The existing HF_TOKEN needs read access to gated public models and must belong to an account granted access to kyutai/pocket-tts. Update the Space secret and restart.') from None
        raise VoiceSetupError('Pocket TTS cloning weights could not download. Check Hugging Face model access and network availability, then retry Play shortly.') from None


def _load_model():
    global _model, _voice_state, _load_error, _retry_after
    if is_loaded():
        return
    if time.monotonic() < _retry_after:
        return
    try:
        import soundfile as sf
        try:
            info = sf.info(str(REFERENCE_AUDIO))
            if info.frames <= 0:
                raise ValueError('Empty reference')
        except Exception as exc:
            raise VoiceSetupError('Jensen reference audio is missing or invalid. Restore jensen_ref.wav, including its Git LFS content.') from exc
        import torch
        from pocket_tts import TTSModel
        torch.set_num_threads(min(2, os.cpu_count() or 1))
        model = TTSModel.load_model(language=LANGUAGE)
        if not model.has_voice_cloning:
            _retry_cloning_download()
            model = TTSModel.load_model(language=LANGUAGE)
        if not model.has_voice_cloning:
            raise VoiceSetupError('Pocket TTS cloning weights are inaccessible. Set the Space HF_TOKEN secret to a read token from an account with access to kyutai/pocket-tts, then restart the Space.')
        state = model.get_state_for_audio_prompt(REFERENCE_AUDIO, truncate=True)
        if not state:
            raise ValueError('No cloned voice state produced')
        # Publish the pair only after reference encoding succeeds. Retry a failed
        # initialization rather than caching an unusable model forever.
        _model, _voice_state = model, state
        _load_error = None
        _retry_after = 0.0
        logger.info('Pocket TTS cloned voice ready: %s / %s', LANGUAGE, REFERENCE_AUDIO.name)
    except Exception as exc:
        _model, _voice_state = None, None
        logger.exception('Pocket TTS cloned voice could not be loaded')
        _load_error = str(exc) if isinstance(exc, VoiceSetupError) else 'Jensen voice could not load. Check Pocket TTS model access and the reference recording, then retry Play shortly.'
        _retry_after = time.monotonic() + 60


def is_available():
    with _lock:
        _load_model()
        return is_loaded()


def get_error():
    return _load_error


def speak(text, output_path=None):
    global _load_error
    if not isinstance(text, str) or not text.strip():
        return None
    with _lock:
        _load_model()
        if not is_loaded():
            return None
        path = None
        try:
            import numpy as np
            import soundfile as sf
            AUDIO_DIR.mkdir(parents=True, exist_ok=True)
            # Each request owns a file: concurrent visitors cannot overwrite a reply.
            path = Path(output_path) if output_path else AUDIO_DIR / f'{uuid.uuid4().hex}.wav'
            cleaned = re.sub(r'[*#`]', '', text).strip()
            if not cleaned:
                return None
            generated = _model.generate_audio(_voice_state, cleaned)
            audio = np.asarray(generated.detach().cpu().numpy(), dtype=np.float32).reshape(-1)
            if not audio.size or not np.isfinite(audio).all():
                raise ValueError('No valid audio generated')
            peak = float(np.max(np.abs(audio)))
            if peak > 0:
                audio = audio / peak * 0.95
            sf.write(str(path), audio, _model.sample_rate, subtype='PCM_16')
            _load_error = None
            cutoff = time.time() - 24 * 60 * 60
            for old in AUDIO_DIR.glob('*.wav'):
                if old != path and old.stat().st_mtime < cutoff:
                    old.unlink(missing_ok=True)
            return str(path)
        except Exception:
            logger.exception('Pocket TTS cloned voice synthesis failed')
            _load_error = 'Could not generate voice. You can still read the reply or retry Play.'
            if path:
                path.unlink(missing_ok=True)
            return None


_voice_slots = threading.BoundedSemaphore(8)


def speak_stream(text, cancel):
    """PCM chunks with real Pocket cancellation; safely serialize this CPU model."""
    import base64
    import numpy as np
    import soundfile as sf
    if not _voice_slots.acquire(blocking=False):
        raise VoiceSetupError('Voice queue is full. Please retry Play shortly.')
    acquired = False
    path = None
    try:
        yield {'type': 'status', 'status': 'queued'}
        while not cancel.is_set():
            if _lock.acquire(timeout=.2): acquired = True; break
        if not acquired or cancel.is_set(): return
        _load_model()
        if not is_loaded(): raise VoiceSetupError(get_error() or 'Voice could not load.')
        if cancel.is_set(): return
        cleaned = re.sub(r'\[\d+\]|[*#`]', '', text).strip()
        if not cleaned: raise VoiceSetupError('Nothing to speak.')
        yield {'type': 'start', 'sample_rate': _model.sample_rate}
        chunks = []
        for generated in _model.generate_audio_stream(_voice_state, cleaned, copy_state=True, stop=cancel):
            if cancel.is_set(): return
            values = np.asarray(generated.detach().cpu().numpy(), dtype=np.float32).reshape(-1)
            if not values.size or not np.isfinite(values).all(): raise ValueError('Invalid audio chunk')
            values = np.clip(values, -.98, .98); chunks.append(values)
            pcm = (values * 32767).astype('<i2').tobytes()
            yield {'type': 'audio', 'pcm': base64.b64encode(pcm).decode(), 'sample_rate': _model.sample_rate}
        if cancel.is_set(): return
        if not chunks: raise VoiceSetupError('Voice generated no audio. Retry Play.')
        AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        path = AUDIO_DIR / f'{uuid.uuid4().hex}.wav'
        sf.write(str(path), np.concatenate(chunks), _model.sample_rate, subtype='PCM_16')
        cutoff = time.time() - 86400
        for old in AUDIO_DIR.glob('*.wav'):
            if old != path and old.stat().st_mtime < cutoff: old.unlink(missing_ok=True)
        yield {'type': 'done', 'audio_file': path.name}
    finally:
        if cancel.is_set() and path: path.unlink(missing_ok=True)
        if acquired: _lock.release()
        _voice_slots.release()
