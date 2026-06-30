# Jensen Huang Digital Twin

A RAG + persona-driven conversational AI that responds as Jensen Huang, built on FastAPI, ChromaDB, Gemini 2.5 Flash, and (optionally) local TTS voice cloning.

**🔗 Live Demo:** _[add your deployed link here]_

---

## Features

- **Persona-locked LLM responses** — strict system prompt enforcing Jensen Huang's voice, length, and tone (`persona.py`)
- **Local, offline RAG retrieval** — ChromaDB + ONNX MiniLM embeddings, zero API calls for retrieval. Implemented Cross Encoder + BiEncoder for more retrieval quality. (`rag.py`, `chroma_utils.py`)
- **Semantic chunking pipeline** — embedding-based chunking instead of naive fixed-size splitting (`semantic_chunker.py`)
- **Long-term memory** — SQLite-backed fact extraction and recall across sessions (`memory.py`)
- **Era filtering** — talk to "2007 Jensen", "2015 Jensen", or present-day Jensen
- **Optional voice cloning** — CPU-friendly TTS via Pocket TTS (`tts.py`)
- **Data pipeline** — YouTube/Whisper transcription + web scraping + cleaning + ingestion (`data_collector.py`, `cleaner.py`, `ingest.py`)
- **Gemini key rotation** — round-robins across multiple API keys on quota errors (`api_client.py`)

---

## Project Structure

```
.
├── main.py                # FastAPI app & API routes
├── agent.py                # Orchestrator: RAG + memory + persona + generation
├── persona.py               # System prompt + keyword-based analogy engine
├── rag.py                   # Query-time retrieval logic
├── chroma_utils.py          # Shared ChromaDB collection/embedding helper
├── semantic_chunker.py      # Embedding-based semantic chunking
├── ingest.py                 # Chunk + embed cleaned transcripts into ChromaDB
├── cleaner.py                 # Strip non-Jensen speech, attach metadata
├── data_collector.py        # Download/transcribe YouTube + scrape web sources
├── memory.py                 # Short-term + long-term (SQLite) memory
├── api_client.py             # Gemini client with key rotation
├── tts.py                     # Optional voice cloning (Pocket TTS)
├── static/
│   └── index.html             # Frontend UI
├── DockerFile
└── requirements.txt
```

---

## Prerequisites

- Python 3.10+ (3.12 recommended — matches Docker base image)
- `ffmpeg` installed on your system (required for audio extraction, only needed if you run the data pipeline or TTS)
- A Google Gemini API key ([Google AI Studio](https://aistudio.google.com/))
- (Optional) A Groq API key for fallback generation
- (Optional) A Hugging Face token if Pocket TTS requires a gated model download

---

## 1. Clone the repository

```bash
git clone <your-repo-url>
cd <your-repo-name>
```

## 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

## 3. Install dependencies

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

> If you don't have a `requirements.txt` yet, you'll need at minimum: `fastapi`, `uvicorn`, `pydantic`, `python-dotenv`, `chromadb`, `sentence-transformers`, `google-genai`, `requests`, `beautifulsoup4`, `pypdf`, `whisper` (openai-whisper), `yt-dlp`, `noisereduce`, `soundfile`, `numpy`, `scipy`.

## 4. Configure environment variables

Create a `.env.local` file in the project root:

```bash
# Gemini keys — at least one of the two patterns below
GOOGLE_API_KEY_1=your_key_here
GOOGLE_API_KEY_2=your_second_key_here   # optional, enables rotation
GOOGLE_API_KEY_3=your_third_key_here    # optional
# OR, single-key fallback:
# GOOGLE_API_KEY=your_key_here

# Optional fallback model if Gemini is rate-limited
GROQ_API_KEY=your_groq_key_here

# Optional, only needed for Pocket TTS gated model downloads
HF_TOKEN=your_huggingface_token_here
```

## 5. Build the knowledge base (optional but recommended)

The repo ships with a data pipeline to scrape and embed Jensen Huang's actual public talks. This step is **only needed once** — the resulting ChromaDB store is reused on every subsequent run.

```bash
# 1. Collect raw transcripts (downloads audio, transcribes with Whisper, scrapes web sources)
python data_collector.py

# 2. Clean transcripts and attach metadata
python cleaner.py

# 3. Chunk + embed into ChromaDB
python ingest.py
```

> Skip this step if you already have a populated `storage/chroma_db/` directory, or if you just want to chat with the persona using general knowledge (no retrieval).

## 6. (Optional) Set up voice cloning

Place a short reference clip named `jensen_ref.wav` in the project root (the data pipeline can auto-generate one — see `extract_reference_clip` in `data_collector.py`). Without this file, TTS is silently disabled and the app still works text-only.

## 7. Run the app

```bash
python main.py
```

Or directly with uvicorn:

```bash
uvicorn main:app --reload --port 8000
```

Visit **http://localhost:8000** in your browser.

## 8. Running with Docker

```bash
docker build -t jensen-digital-twin -f DockerFile .
docker run -p 8000:8000 --env-file .env.local jensen-digital-twin
```

> ⚠️ Double-check the exposed/published port against what's actually in the `DockerFile` `CMD` before running this (see Known Issues below) — there's a mismatch between the documented port and the one uvicorn binds to.

---

## API Endpoints

| Method | Route | Description |
|---|---|---|
| `GET` | `/` | Serves the frontend |
| `POST` | `/api/chat` | Send a message, get persona response (+ optional audio) |
| `POST` | `/api/greet` | Generate a personalized greeting |
| `POST` | `/api/speak` | Generate TTS audio for arbitrary text |
| `GET` | `/api/audio/{filename}` | Fetch generated audio file |
| `POST` | `/api/session/save` | Extract & persist long-term memories from current session |
| `POST` | `/api/session/reset` | Clear short-term conversation history |
| `GET` | `/api/memory/{user_id}` | Fetch stored long-term memories |
| `DELETE` | `/api/memory/{user_id}` | Delete all long-term memories for a user |

---

## Known Issues / Bugs

See the full bug report below — fix these before treating this as production-ready.

---

## License

Add your license here.

---

## Improvements

- Try to implement colBERT Mechanism for rag
- Using a pipeline to find most common words used in transcripts to generate persona
- RAG Architecture to be made more complex(for example Pixel RAG)