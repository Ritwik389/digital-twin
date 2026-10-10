"""Bounded provider retries and a current Groq fallback, shared by chat and memory."""
import json
import logging
import os
import re
import threading
import time

import requests
from dotenv import load_dotenv

load_dotenv('.env.local')
logger = logging.getLogger(__name__)
MODEL = os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')
_lock = threading.Lock()
_current_index = 0
RETRYABLE = {408, 429, 500, 502, 503, 504}


class GenerationError(RuntimeError):
    """Safe, user-visible error that never includes credentials or request URLs."""


def _collect_keys():
    keys = [os.getenv(f'GOOGLE_API_KEY_{i}', '').strip() for i in range(1, 4)]
    keys += [os.getenv(name, '').strip() for name in ('GOOGLE_API_KEY', 'GEMINI_API_KEY')]
    return list(dict.fromkeys(key for key in keys if key))


def _request(url, *, headers, payload, attempts=2):
    for attempt in range(attempts):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=(5, 25))
        except requests.RequestException:
            if attempt + 1 == attempts:
                raise GenerationError('Generation service timed out. Please try again.') from None
        else:
            if response.ok:
                try:
                    return response.json()
                except ValueError:
                    raise GenerationError('Generation service returned an invalid response.') from None
            if response.status_code not in RETRYABLE or attempt + 1 == attempts:
                raise GenerationError(f'Generation service returned HTTP {response.status_code}.')
        time.sleep(0.5 * (2 ** attempt))


def gemini_generate(prompt, model=None):
    global _current_index
    keys = _collect_keys()
    if not keys:
        raise GenerationError('Gemini is not configured.')
    with _lock:
        start = _current_index % len(keys)
        _current_index += 1
    last_error = None
    # Rotate configured keys, and retry transient failures even with only one key.
    for offset in range(len(keys)):
        try:
            data = _request(
                f'https://generativelanguage.googleapis.com/v1beta/models/{(model or MODEL).removeprefix("models/")}:generateContent',
                headers={'x-goog-api-key': keys[(start + offset) % len(keys)]},
                payload={'contents': [{'role': 'user', 'parts': [{'text': prompt}]}]},
            )
            parts = data.get('candidates', [{}])[0].get('content', {}).get('parts', [])
            text = ''.join(p.get('text', '') for p in parts if not p.get('thought')).strip()
            if not text:
                raise GenerationError('Gemini returned no answer.')
            return text
        except (GenerationError, IndexError, TypeError) as exc:
            last_error = exc
    raise GenerationError(str(last_error))


def groq_generate(prompt):
    key = os.getenv('GROQ_API_KEY', '').strip()
    if not key:
        raise GenerationError('Groq is not configured.')
    data = _request(
        'https://api.groq.com/openai/v1/chat/completions',
        headers={'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'},
        payload={
            'model': os.getenv('GROQ_MODEL', 'openai/gpt-oss-120b'),
            'messages': [{'role': 'user', 'content': prompt}],
            'temperature': 0.7,
            'max_completion_tokens': 2048,
        },
    )
    try:
        text = data['choices'][0]['message']['content'].strip()
    except (KeyError, IndexError, TypeError, AttributeError):
        raise GenerationError('Groq returned no answer.') from None
    if not text:
        raise GenerationError('Groq returned no answer.')
    return text


def generate_text(prompt):
    if not _collect_keys() and not os.getenv('GROQ_API_KEY', '').strip():
        raise GenerationError('Set GOOGLE_API_KEY (or GOOGLE_API_KEY_1) or GROQ_API_KEY in Space secrets.')
    if _collect_keys():
        try:
            return gemini_generate(prompt)
        except GenerationError as exc:
            logger.warning('Gemini unavailable: %s; trying Groq', exc)
    try:
        return groq_generate(prompt)
    except GenerationError:
        raise GenerationError('Chat service is temporarily unavailable. Please try again in a moment.') from None


def parse_json(text):
    text = re.sub(r'^```(?:json)?\s*|\s*```$', '', text.strip())
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return None


def get_key_count():
    return len(_collect_keys())


class GenerationCancelled(Exception):
    pass


def _sse_stream(url, headers, payload, provider, cancel):
    """Retry only before the first token. Never mix providers after partial output."""
    started = False
    for attempt in range(2):
        response = None
        retry_delay = .5
        try:
            if cancel.is_set(): raise GenerationCancelled()
            response = requests.post(url, headers=headers, json=payload, stream=True, timeout=(5, 25))
            if not response.ok:
                logger.warning('%s stream returned HTTP %s', provider, response.status_code)
                if response.status_code == 429:
                    try: retry_delay = max(.5, min(60, float(getattr(response, 'headers', {}).get('retry-after', '.5'))))
                    except (TypeError, ValueError): pass
                raise GenerationError(f'Generation service returned HTTP {response.status_code}.')
            response.encoding = 'utf-8'
            completed = False
            for line in response.iter_lines(decode_unicode=True, chunk_size=256):
                if cancel.is_set(): raise GenerationCancelled()
                if not line or not line.startswith('data:'): continue
                raw = line[5:].strip()
                if raw == '[DONE]': completed = True; break
                item = json.loads(raw)
                if item.get('error'): raise GenerationError('Generation stream failed.')
                if provider == 'gemini':
                    candidate = (item.get('candidates') or [{}])[0]
                    text = ''.join(p.get('text', '') for p in candidate.get('content', {}).get('parts', []) if not p.get('thought'))
                    if candidate.get('finishReason') == 'STOP': completed = True
                    elif candidate.get('finishReason'): raise GenerationError('Generation did not complete.')
                else:
                    choice = (item.get('choices') or [{}])[0]
                    text = choice.get('delta', {}).get('content') or ''
                    if choice.get('finish_reason') == 'stop': completed = True
                    elif choice.get('finish_reason'): raise GenerationError('Generation did not complete.')
                if text: started = True; yield text
            if not started or not completed: raise GenerationError('Generation stream ended before the answer completed.')
            return
        except GenerationCancelled: raise
        except (requests.RequestException, ValueError, TypeError, GenerationError) as exc:
            if started:
                raise GenerationError('Response interrupted. Please retry.') from None
            if attempt == 1:
                if isinstance(exc, GenerationError) and 'HTTP 429' in str(exc):
                    raise GenerationError('The chat provider is busy. Please try again shortly.') from None
                raise GenerationError('The chat provider could not start a response. Please retry.') from None
            if isinstance(exc, GenerationError) and 'HTTP' in str(exc):
                match = re.search(r'HTTP (\d+)', str(exc))
                if match and int(match[1]) not in RETRYABLE: raise
            if response is not None: response.close()
            if cancel.wait(retry_delay): raise GenerationCancelled()
        finally:
            if response is not None: response.close()


def generate_text_stream(prompt, cancel=None):
    cancel = cancel or threading.Event()
    keys = _collect_keys()
    config = {'maxOutputTokens': 4096}
    if MODEL.removeprefix('models/').startswith('gemini-2.5-flash'):
        config['thinkingConfig'] = {'thinkingBudget': 256}
    started = False
    global _current_index
    if keys:
        with _lock:
            start = _current_index % len(keys); _current_index += 1
        for offset in range(len(keys)):
            try:
                for piece in _sse_stream(
                    f'https://generativelanguage.googleapis.com/v1beta/models/{MODEL.removeprefix("models/")}:streamGenerateContent?alt=sse',
                    {'x-goog-api-key': keys[(start + offset) % len(keys)]},
                    {'contents': [{'role': 'user', 'parts': [{'text': prompt}]}], 'generationConfig': config}, 'gemini', cancel):
                    started = True; yield piece
                return
            except GenerationError:
                if started: raise
                logger.warning('Gemini stream unavailable; trying the next configured provider')
    key = os.getenv('GROQ_API_KEY', '').strip()
    if not key: raise GenerationError('Chat service is temporarily unavailable. Please try again.')
    yield from _sse_stream('https://api.groq.com/openai/v1/chat/completions',
        {'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'},
        {'model': os.getenv('GROQ_MODEL', 'openai/gpt-oss-120b'), 'messages': [{'role': 'user', 'content': prompt}],
         'temperature': .6, 'max_completion_tokens': 1800, 'stream': True}, 'groq', cancel)
