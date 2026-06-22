"""
cleaner.py
Strips non-Jensen speech from raw transcripts.
Adds metadata headers.
Outputs cleaned .txt files to data/cleaned/
"""

from pathlib import Path
import re

RAW_DIR = Path("data/raw")
CLEAN_DIR = Path("data/cleaned")
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

NON_JENSEN_LABELS = [
    "Ben:", "David:", "Lex:", "Interviewer:", "Host:", "Moderator:",
    "Audience:", "Question:", "Q:", "Speaker:", "Ben Horowitz:",
    "David Rosenthal:", "Lex Fridman:", "Student:",
]

SOURCE_METADATA = {
    "gtc_2019": {"source": "gtc_keynote", "year": 2019, "domain": "gpu_computing"},
    "gtc_2020": {"source": "gtc_keynote", "year": 2020, "domain": "gpu_computing"},
    "gtc_2021": {"source": "gtc_keynote", "year": 2021, "domain": "gpu_computing"},
    "gtc_2022": {"source": "gtc_keynote", "year": 2022, "domain": "ai_infrastructure"},
    "gtc_2023": {"source": "gtc_keynote", "year": 2023, "domain": "ai_infrastructure"},
    "gtc_2024": {"source": "gtc_keynote", "year": 2024, "domain": "ai_infrastructure"},
    "acquired_2023": {"source": "acquired_podcast", "year": 2023, "domain": "business_philosophy"},
    "lex_2023": {"source": "lex_podcast", "year": 2023, "domain": "personal_philosophy"},
    "stanford_gsb_2023": {"source": "stanford_interview", "year": 2023, "domain": "business_philosophy"},
    "investor_day_2022": {"source": "investor_day", "year": 2022, "domain": "chip_architecture"},
    "investor_day_2023": {"source": "investor_day", "year": 2023, "domain": "ai_infrastructure"},
    "gtc_2025": {"source": "gtc_keynote", "year": 2025, "domain": "ai_infrastructure"},
    "gtc_2026": {"source": "gtc_keynote", "year": 2026, "domain": "gpu_computing"},
    "gtc_spring_2021": {"source": "gtc_keynote", "year": 2021, "domain": "gpu_computing"},
    "gtc_spring_2022": {"source": "gtc_keynote", "year": 2022, "domain": "ai_infrastructure"},
    "keynote_2024": {"source": "gtc_keynote", "year": 2024, "domain": "business_philosophy"},
    "lex_2026": {"source": "lex_podcast", "year": 2026, "domain": "business_philosophy"},
    "milken_2024": {"source": "milken_2024", "year": 2024, "domain": "ai_infrastructure"},
}


def strip_non_jensen_labeled(text: str) -> str:
    lines = text.split("\n")
    jensen_lines = []
    capturing = True

    for line in lines:
        line = line.strip()
        if not line:
            continue
        is_other_speaker = any(line.startswith(label) for label in NON_JENSEN_LABELS)
        is_jensen = line.startswith("Jensen:") or line.startswith("JENSEN:")

        if is_jensen:
            capturing = True
            line = re.sub(r"^(Jensen:|JENSEN:)\s*", "", line)
        elif is_other_speaker:
            capturing = False
            continue

        if capturing and line:
            jensen_lines.append(line)

    return "\n".join(jensen_lines)


def clean_keynote(text: str) -> str:
    # Keynotes are mostly Jensen — remove common non-speech artifacts
    text = re.sub(r"\[.*?\]", "", text)           # [applause], [music], etc.
    text = re.sub(r"\(.*?\)", "", text)           # (audience laughs)
    text = re.sub(r"<[^>]+>", "", text)           # any HTML tags
    text = re.sub(r"\n{3,}", "\n\n", text)        # collapse excessive newlines
    text = re.sub(r"[ \t]{2,}", " ", text)        # collapse multiple spaces
    return text.strip()


def write_cleaned(name: str, text: str, meta: dict):
    out_path = CLEAN_DIR / f"{name}.txt"
    header = (
        f"SOURCE: {meta['source']}\n"
        f"YEAR: {meta['year']}\n"
        f"DOMAIN: {meta['domain']}\n"
        f"---\n"
    )
    out_path.write_text(header + text, encoding="utf-8")
    print(f"[cleaned] {name}.txt ({len(text.split())} words)")


def clean_all():
    for raw_file in sorted(RAW_DIR.glob("*_raw.txt")):
        name = raw_file.stem.replace("_raw", "")
        if name not in SOURCE_METADATA:
            print(f"[skip] no metadata for {name}")
            continue

        meta = SOURCE_METADATA[name]
        raw_text = raw_file.read_text(encoding="utf-8")

        if meta["source"] in ("acquired_podcast", "lex_podcast", "stanford_interview"):
            cleaned = strip_non_jensen_labeled(raw_text)
        else:
            cleaned = clean_keynote(raw_text)

        if len(cleaned.split()) < 100:
            print(f"[warn] {name} has fewer than 100 words after cleaning — check source")
            continue

        write_cleaned(name, cleaned, meta)

    print(f"\n[done] Cleaned files in data/cleaned/")


if __name__ == "__main__":
    clean_all()
