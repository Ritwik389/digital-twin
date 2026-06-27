---
title: Digital Twin
emoji: 🚀
colorFrom: pink
colorTo: yellow
sdk: gradio
sdk_version: 6.17.3
python_version: '3.11'
app_file: app.py
pinned: false

---

# Jensen Huang Digital Twin — Basic Edition

A simplified, dependency-light version of the project: basic RAG over
cleaned transcripts, short-term + long-term memory, a Jensen Huang
persona, a Gradio chat UI, and optional voice cloning. voice cloning  uses
Qwen3-TTS 

## What changed

| File | Status | Notes |
|---|---|---|
| `data_collector.py` |  | downloads/transcribes/scrapes raw transcripts, extracts `jensen_ref.wav` |
| `cleaner.py` |  | strips non-Jensen speech, writes `data/cleaned/*.txt` |
| `ingest.py` | Semantic Chunking using all-MiniLM-L6-v2 | chunks + stores in ChromaDB; embeddings are now computed locally via ChromaDB's built-in ONNX MiniLM model instead of the Gemini embedding API |
| `chroma_utils.py` | | shared helper so `ingest.py` and `rag.py` use the exact same offline embedding function |
| `rag.py`  | basic top-k retrieval only — no LangGraph, no query rewriting/decomposition/critic loop, no KG extraction, zero API calls |
| `api_client.py` | | only `gemini_generate` (final answer) + `parse_json`; embedding helpers removed |
| `memory.py` | | short-term window + SQLite long-term memory; KG-triple conversion removed |
| `persona.py` | | system prompt + analogy bank; unused reasoning-trace builder removed |
| `agent.py` | | orchestrator: retrieve (local) → memory (local) → persona → **one** Gemini call |
| `app.py`  | replaces the Streamlit UI; chat + sidebar (era filter, sources, long-term memory viewer, save/reset, voice toggle) |
| `tts.py` | voice cloning from `jensen_ref.wav` using `Qwen/Qwen3-TTS-12Hz-0.6B-Base` |

## API call budget

- **Ingestion**: 
- **Per chat turn**: exactly **1** Gemini call (`gemini_generate`, the final answer).
  Retrieval, long-term-memory lookup, and analogy selection are all local.
- **"Save Session" button**: 1 extra Gemini call to extract long-term-memory
  facts from the conversation (only when clicked, not per turn).

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env.local` file with at least one Gemini API key:

```
GOOGLE_API_KEY=your_key_here
# optional extra keys for round-robin rotation on rate limits:
# GOOGLE_API_KEY_1=...
# GOOGLE_API_KEY_2=...
# GOOGLE_API_KEY_3=...
```

## Pipeline (run once / when adding new sources)

```bash
python data_collector.py   # -> data/raw/*_raw.txt
python cleaner.py           # -> data/cleaned/*.txt
python ingest.py            # -> ./chroma_db (local vector store)
```

`ingest.py` only needs to be re-run when `data/cleaned/` changes — it
skips files already present in ChromaDB. The first run downloads a
small (~80 MB) ONNX embedding model and caches it locally; after that,
everything runs fully offline.

## Run the app

```bash
python app.py
```

This starts a Gradio 6 app (default at `http://127.0.0.1:7860`) with:

- **Chat panel** — talk to Jensen, with per-turn badges showing whether
  retrieval happened and how many chunks were used.
- **Sidebar** — set a User ID, pick an era filter (All / Pre-CUDA /
  Deep Learning / LLM Era), view the sources retrieved for the last
  turn, and load/clear long-term memory facts for that user.
- **Voice toggle** — only appears if `jensen_ref.wav` exists. When
  enabled, every answer is also synthesized in Jensen's cloned voice
  and played back via the audio player.
- **Save Session** — extracts durable facts about the user from the
  conversation into SQLite (`data/memory.db`) for future sessions.
- **Reset Chat** — clears the short-term conversation window.

## Voice cloning (optional)

`tts.py` clones Jensen's voice from `jensen_ref.wav` (extracted by
`data_collector.py`) using **Pocket-TTS** (`Pocket TTS`
by default — Apache-2.0, ~0.6B params).

- For the best quality, put a transcript of `jensen_ref.wav` in
  `jensen_ref.txt`. If that file doesn't exist, `tts.py` transcribes
  the clip once with Whisper and caches the result. If Whisper isn't
  available either, it falls back to x-vector-only cloning (no
  transcript needed, slightly lower fidelity).
- The reference features ("clone prompt") are computed **once** at
  first use and reused for every line Jensen speaks afterward.
- This feature is fully optional: if `qwen-tts`/`torch` aren't
  installed, or `jensen_ref.wav` is missing, the voice toggle simply
  doesn't appear and the rest of the app works normally.
- First run downloads the Qwen3-TTS model weights (a few GB) from
  Hugging Face — needs internet access once, then it's cached locally.
- See `requirements.txt` for CPU-only vs GPU install instructions —
  on a CPU-only machine, install `torch` from the CPU wheel index
  first to avoid pulling several GB of unused CUDA packages.

## Architecture

```
User query
   │
   ▼
rag.py  ── local ONNX embedding ──► ChromaDB (chroma_db/)
   │  (top-k chunks, or "no_retrieval" for greetings)
   ▼
memory.py ── SQLite keyword lookup ──► long-term facts (data/memory.db)
   │
   ▼
persona.py ── builds system prompt + (optional) analogy
   │
   ▼
agent.py ── assembles final prompt
   │
   ▼
api_client.py ── gemini_generate() ──► Gemini 2.5 Flash   (the ONE API call)
   │
   ▼
app.py (html js css) ── displays answer + sources + memory
```
- Added Jailbreaking via a prompt

## Future Improvements

- Using a pipeline to find most common words used in transcripts to generate persona
- RAG Architecture to be made more complex(for example Pixel RAG)
- Deploying
- fall back model like ollama when gemini hits ratelimits
- proper description of Jensen in the front page
- a landing page to input the user's name and id
- fix the time aware context , and allowing users to start a new session while storing the chat log for each user session till they don't reload as otherwise database will need to be created and it will be complicated


