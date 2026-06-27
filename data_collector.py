"""
data_collector.py
Downloads YouTube audio, transcribes with Whisper, scrapes web transcripts.
Outputs raw .txt files to data/raw/
"""

import os
import subprocess
import whisper
import requests
from bs4 import BeautifulSoup
from pathlib import Path

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

YOUTUBE_SOURCES = [
    {
        "url": "https://www.youtube.com/watch?v=Y2F8yisiS6E",
        "name": "gtc_2024",
        "year": 2024,
        "domain": "ai_infrastructure",
    },
    {
        "url": "https://www.youtube.com/watch?v=cEg8cOx7UZk",
        "name": "keynote_2024",
        "year": 2024,
        "domain": "business_philosophy",
    },
    {
        "url": "https://www.youtube.com/watch?v=11Y3B33oCLE",
        "name": "gtc_2026",
        "year": 2026,
        "domain": "gpu_computing",
    },
    {
        "url": "https://www.youtube.com/watch?v=_waPvOwL9Z8",
        "name": "gtc_2025",
        "year": 2025,
        "domain": "ai_infrastructure",
    },
    {
        "url": "https://www.youtube.com/watch?v=DiGB5uAYKAg",
        "name": "gtc_2023",
        "year": 2023,
        "domain": "ai_infrastructure",
    },
    {
        "url": "https://www.youtube.com/watch?v=Z2XlNfCtxwI",
        "name": "gtc_2019",
        "year": 2019,
        "domain": "ai_infrastructure",
    },
    {
        "url": "https://www.youtube.com/watch?v=39ubNuxnrK8",
        "name": "gtc_spring_2022",
        "year": 2022,
        "domain": "ai_infrastructure",
    },
    {
        "url": "https://www.youtube.com/watch?v=PWcNlRI00jo",
        "name": "gtc_2022",
        "year": 2022,
        "domain": "robotics",
    },
    {
        "url": "https://www.youtube.com/watch?v=eAn_oiZwUXA",
        "name": "gtc_spring_2021",
        "year": 2021,
        "domain": "gpu_computing",
    },
    {
        "url": "https://www.youtube.com/watch?v=jhDiaUL_RaM",
        "name": "gtc_2021",
        "year": 2021,
        "domain": "robotics",
    },
]

WEB_SOURCES = [
    {
        "url": "https://lexfridman.com/jensen-huang-transcript/",
        "name": "lex_2026",
        "year": 2026,
        "domain": "business_philosophy",
        "speaker_label": "Jensen Huang",
    },
    {
        "url": "https://milkeninstitute.org/sites/default/files/2025-05/new-innovation-economy-conversation-nvidia-ceo-jensen-huang_Transcript_GC25.pdf",
        "name": "milken_2026",
        "year": 2026,
        "domain": "business_philosophy",
        "speaker_label": "Jensen Huang",
    },
    {
        "url": "https://en.wikipedia.org/wiki/Jensen_Huang",
        "name": "jensen_2026",
        "year": 2026,
        "domain": "business_philosophy",
        "speaker_label": None,
    },

]


def download_audio(url: str, name: str) -> Path:
    out_path = RAW_DIR / f"{name}.wav"
    if out_path.exists():
        print(f"[skip] {name}.wav already exists")
        return out_path
    print(f"[download] {name}")
    subprocess.run([
        "yt-dlp",
        "-x", "--audio-format", "wav",
        "--audio-quality", "0",
        "-o", str(RAW_DIR / f"{name}.%(ext)s"),
        url
    ], check=True)
    return out_path


def clean_audio(wav_path: Path) -> Path:
    import noisereduce as nr
    import soundfile as sf
    import numpy as np

    cleaned_path = RAW_DIR / f"{wav_path.stem}_clean.wav"
    if cleaned_path.exists():
        print(f"[skip] {cleaned_path.name} already cleaned")
        return cleaned_path

    print(f"[clean audio] {wav_path.name}")
    data, rate = sf.read(str(wav_path))
    if data.ndim > 1:
        data = data[:, 0]
    reduced = nr.reduce_noise(y=data, sr=rate, stationary=False)
    sf.write(str(cleaned_path), reduced, rate)
    return cleaned_path


def extract_reference_clip(wav_path: Path, start_sec: int = 2519, duration: int = 10) -> Path:
    ref_path = Path("jensen_ref.wav")
    if ref_path.exists():
        print("[skip] jensen_ref.wav already exists")
        return ref_path
    print("[extract] reference clip for XTTS voice cloning")
    subprocess.run([
        "ffmpeg", "-y",
        "-ss", str(start_sec),
        "-t", str(duration),
        "-i", str(wav_path),
        "-af", "loudnorm",
        "-ar", "22050",
        "-ac", "1",
        str(ref_path)
    ], check=True)
    return ref_path


def transcribe(wav_path: Path, name: str) -> Path:
    txt_path = RAW_DIR / f"{name}_raw.txt"
    if txt_path.exists():
        print(f"[skip] {name}_raw.txt already transcribed")
        return txt_path
    print(f"[transcribe] {name}")
    model = whisper.load_model("base")
    result = model.transcribe(str(wav_path))
    txt_path.write_text(result["text"], encoding="utf-8")
    return txt_path



import io
import requests
from pathlib import Path
from bs4 import BeautifulSoup
from pypdf import PdfReader


def scrape_web_transcript(url: str, name: str, speaker_label: str) -> Path:
    txt_path = RAW_DIR / f"{name}_raw.txt"

    if txt_path.exists():
        print(f"[skip] {txt_path.name} already scraped")
        return txt_path

    print(f"[scrape] {name}")

    try:
        resp = requests.get(
            url,
            timeout=30,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/125.0 Safari/537.36"
                )
            },
        )

        resp.raise_for_status()

    except requests.exceptions.Timeout:
        print(f"[error] Timeout while fetching {url}")
        raise

    except requests.exceptions.RequestException as e:
        print(f"[error] Failed to fetch {url}")
        print(e)
        raise

    try:
        content_type = resp.headers.get("Content-Type", "").lower()


        if (
            "application/pdf" in content_type
            or url.lower().endswith(".pdf")
        ):
            print("[info] PDF detected")

            pdf_reader = PdfReader(io.BytesIO(resp.content))

            pages = []

            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text() or ""
                    pages.append(page_text)

                except Exception as e:
                    print(
                        f"[warn] Failed extracting page {page_num + 1}: {e}"
                    )

            full_text = "\n\n".join(pages)

        else:
            print("[info] HTML page detected")

            soup = BeautifulSoup(resp.text, "html.parser")

            paragraphs = [
                p.get_text(separator=" ", strip=True)
                for p in soup.find_all("p")
            ]

            full_text = "\n".join(paragraphs)
        if not full_text.strip():
            raise ValueError("No text extracted from source")

        txt_path.write_text(full_text, encoding="utf-8")

        print(
            f"[saved] {txt_path.name} "
            f"({len(full_text):,} characters)"
        )

        return txt_path

    except Exception as e:
        print(f"[error] Failed to process content from {url}")
        print(e)
        raise



def collect_all():
    ref_wav = None
    for src in YOUTUBE_SOURCES:
        wav = download_audio(src["url"], src["name"])
        clean = clean_audio(wav)
        transcribe(clean, src["name"])
        if ref_wav is None:
            ref_wav = clean

    if ref_wav:
        extract_reference_clip(ref_wav)

    for src in WEB_SOURCES:
        scrape_web_transcript(src["url"], src["name"], src["speaker_label"])

    print("\n[done] All raw transcripts in data/raw/")


if __name__ == "__main__":
    collect_all()
