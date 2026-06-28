

import re
import json
import sqlite3
from pathlib import Path

from api_client import gemini_generate

DB_PATH = Path("./storage/memory.db")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

MAX_HISTORY = 16  # 8 turns x 2 (user + assistant)

EXTRACTION_PROMPT = """You are analyzing a conversation between a user and Jensen Huang (digital twin).
Extract important facts about the USER that should be remembered for future conversations.
Focus on: user's background, interests, expertise level, questions they care about, projects they mentioned.
Do NOT extract facts about Jensen or NVIDIA — only about the user.

Return ONLY a JSON array of objects. Each object has:
  "fact": string (the fact to remember),
  "category": one of "background" | "interest" | "expertise" | "project" | "preference" | "general",
  "importance": integer 1-3 (3 = most important)

If nothing worth remembering, return [].
No markdown, no explanation, just the JSON array.

Conversation:
{conversation}"""

class ShortTermMemory:
    def __init__(self, window: int = 8):
        self.window = window
        self.history: list[dict] = []

    def add(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
        if len(self.history) > self.window * 2:
            self.history = self.history[-(self.window * 2):]

    def get_history(self) -> list[dict]:
        return list(self.history)

    def format_for_prompt(self) -> str:
        if not self.history:
            return ""
        lines = []
        for turn in self.history:
            label = "User" if turn["role"] == "user" else "Jensen"
            lines.append(f"{label}: {turn['content']}")
        return "\n".join(lines)

    def turn_count(self) -> int:
        return len(self.history) // 2

    def clear(self):
        self.history = []


class LongTermMemory:
    """
    Persists facts across sessions in SQLite.
    """

    def __init__(self, user_id: str):
        self.user_id = user_id
        self._init_db()

    def _open(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        conn = self._open()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS long_term_memory (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id     TEXT NOT NULL,
                    fact        TEXT NOT NULL,
                    category    TEXT DEFAULT 'general',
                    importance  INTEGER DEFAULT 1,
                    created_at  TEXT DEFAULT (datetime('now'))
                )
            """)
            conn.commit()
        finally:
            conn.close()

    def save_facts(self, facts: list[dict]):
        if not facts:
            return
        conn = self._open()
        try:
            for f in facts:
                fact_text = f.get("fact", "").strip()
                if not fact_text:
                    continue
                conn.execute(
                    "INSERT INTO long_term_memory (user_id, fact, category, importance) VALUES (?,?,?,?)",
                    (
                        self.user_id,
                        fact_text,
                        f.get("category", "general"),
                        f.get("importance", 1),
                    ),
                )
            conn.commit()
        finally:
            conn.close()

    def load_all(self) -> list[dict]:
        conn = self._open()
        try:
            rows = conn.execute(
                "SELECT fact, category, importance, created_at FROM long_term_memory "
                "WHERE user_id = ? ORDER BY importance DESC, created_at DESC",
                (self.user_id,)
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def load_relevant(self, query: str, limit: int = 6) -> list[dict]:
        """Keyword overlap scoring — no embedding/API call needed."""
        words = set(re.findall(r"\b[a-z]{4,}\b", query.lower()))
        all_facts = self.load_all()

        if not words:
            return all_facts[:limit]

        scored = []
        for f in all_facts:
            hits = sum(1 for w in words if w in f["fact"].lower())
            scored.append((hits, f["importance"], f))

        scored.sort(key=lambda x: (-x[0], -x[1]))
        relevant = [f for hits, _, f in scored if hits > 0]

        # Pad with top facts if not enough relevant ones
        if len(relevant) < limit:
            seen = {f["fact"] for f in relevant}
            for f in all_facts:
                if f["fact"] not in seen:
                    relevant.append(f)
                if len(relevant) >= limit:
                    break

        return relevant[:limit]

    def format_for_prompt(self, facts: list[dict]) -> str:
        if not facts:
            return ""
        lines = ["Things Jensen remembers about this person from past conversations:"]
        for f in facts:
            lines.append(f"- [{f['category']}] {f['fact']}")
        return "\n".join(lines)

    def delete_fact(self, fact_text: str):
        conn = self._open()
        try:
            conn.execute(
                "DELETE FROM long_term_memory WHERE user_id = ? AND fact = ?",
                (self.user_id, fact_text)
            )
            conn.commit()
        finally:
            conn.close()

    def clear_all(self):
        conn = self._open()
        try:
            conn.execute("DELETE FROM long_term_memory WHERE user_id = ?", (self.user_id,))
            conn.commit()
        finally:
            conn.close()



def extract_memories(conversation: list[dict]) -> list[dict]:

    if len(conversation) < 4:
        return []

    conv_text = "\n".join(
        f"{t['role'].upper()}: {t['content'][:300]}"
        for t in conversation
    )
    prompt = EXTRACTION_PROMPT.format(conversation=conv_text)

    try:
        raw = gemini_generate(prompt)
        raw = re.sub(r"```json|```", "", raw).strip()
        facts = json.loads(raw)
        if isinstance(facts, list):
            return facts
    except Exception as e:
        print(f"[memory] extraction failed: {e}")

    return []


def get_relevant_memories(user_id: str, query: str) -> list[str]:
    mem = LongTermMemory(user_id)
    facts = mem.load_relevant(query)
    return [f["fact"] for f in facts]


def get_all_memories(user_id: str) -> list[dict]:
    return LongTermMemory(user_id).load_all()


def save_session_memories(user_id: str, conversation: list[dict]):

    facts = extract_memories(conversation)
    if facts:
        LongTermMemory(user_id).save_facts(facts)
        print(f"[memory] saved {len(facts)} facts for user {user_id}")
    else:
        print("[memory] no new facts extracted")


def delete_all_memories(user_id: str):
    LongTermMemory(user_id).clear_all()


def format_memories_for_prompt(user_id: str, query: str) -> str:
    mem = LongTermMemory(user_id)
    facts = mem.load_relevant(query)
    return mem.format_for_prompt(facts)


if __name__ == "__main__":
    # Quick test
    stm = ShortTermMemory(window=3)
    stm.add("user", "What is CUDA?")
    stm.add("assistant", "CUDA is our parallel computing platform...")
    stm.add("user", "When was it released?")
    print("Short-term:\n", stm.format_for_prompt())

    ltm = LongTermMemory("test_user")
    ltm.save_facts([
        {"fact": "User is a DTU student working on AI projects", "category": "background", "importance": 3},
        {"fact": "User is interested in GPU computing and CUDA", "category": "interest", "importance": 2},
    ])
    print("\nLong-term (all):")
    for f in ltm.load_all():
        print(f" - [{f['category']}] {f['fact']}")

    print("\nRelevant to 'GPU architecture':")
    for f in ltm.load_relevant("GPU architecture"):
        print(f" - {f['fact']}")
