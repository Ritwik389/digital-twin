"""
agent.py
Jensen Huang Digital Twin — main orchestrator.
Combines: basic RAG (rag.py) + memory (memory.py) + persona (persona.py)



API call flow:
  - Retrieval (rag.py) is 100% local/offline — zero API calls.
  - Long-term memory lookup is keyword-scored SQLite — zero API calls.
  - Analogy injection is keyword-matched — zero API calls.
  - Exactly ONE Gemini call per chat turn: the final answer generation.
"""



from rag import retrieve
from memory import ShortTermMemory, get_relevant_memories, save_session_memories
from persona import build_system_prompt, get_analogy_for_concept
from api_client import gemini_generate
import requests


ERA_YEAR_RANGES = {
    "all": (None, None),
    "pre_cuda": (None, 2007),
    "deep_learning": (2012, 2021),
    "llm_era": (2022, None),
}


class JensenAgent:
    def __init__(self, user_id: str = "default_user"):
        self.user_id = user_id
        self.short_term = ShortTermMemory(window=8)
        self.era_filter = "all"

    def set_era(self, era: str):
        if era in ERA_YEAR_RANGES:
            self.era_filter = era

    def chat(self, query: str) -> dict:
        year_min, _ = ERA_YEAR_RANGES[self.era_filter]
        conversation_history = self.short_term.format_for_prompt()

        # 1. Basic RAG (local, offline embeddings — no API call)
        rag_result = retrieve(
            query=query,
            year_filter=year_min,
            domain_filter=None,
            conversation_history=conversation_history,
        )

        # 2. Long-term memory — keyword-scored, no API call
        long_term_memories: list[str] = []
        long_term_prompt_block = ""
        if rag_result["needs_retrieval"]:
            long_term_memories = get_relevant_memories(self.user_id, query)
            if long_term_memories:
                lines = ["Things Jensen remembers about this person:"]
                lines += [f"- {m}" for m in long_term_memories]
                long_term_prompt_block = "\n".join(lines)

        # 3. System prompt with era + long-term memories
        system_prompt = build_system_prompt(
            long_term_memories=long_term_memories,
            era_filter=self.era_filter,
        )

        # 4. Analogy injection (keyword-based, no API call)
        analogy = get_analogy_for_concept(query) if rag_result["needs_retrieval"] else None

        # 5. Build full generation prompt
        full_prompt = self._build_prompt(
            query=query,
            context=rag_result["context"],
            analogy=analogy,
            conversation_history=conversation_history,
            long_term_prompt_block=long_term_prompt_block,
            system_prompt=system_prompt,
            query_type=rag_result["query_type"],
        )
        try:
            # 6. Generate (the ONE Gemini call per query)
            answer = gemini_generate(full_prompt)
        except Exception as e:
            # Catches the 503 ServerError and network timeouts
            print(f"\n[⚠️ API ERROR] Gemini generation failed: {e}")
            print("[🔄 FALLBACK] Routing query to free Groq Cloud (Llama 3)...")
            
            try:
                # Get your free Groq key from environment variables (or paste it here for testing)
                groq_key = os.environ.get("GROQ_API_KEY", "your_free_groq_api_key_here")
                
                headers = {
                    "Authorization": f"Bearer {groq_key}",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "model": "llama-3.3-70b-versatile", # Groq's fast, free open-source model
                    "messages": [{"role": "user", "content": full_prompt}],
                    "temperature": 0.7
                }
                
                # Groq uses standard OpenAI-compatible HTTP requests
                response = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions", 
                    headers=headers, 
                    json=payload, 
                    timeout=15 
                )
                response.raise_for_status()
                answer = response.json()["choices"][0]["message"]["content"]
                
            except Exception as fallback_error:
                print(f"[🚨 CRITICAL] Groq fallback also failed: {fallback_error}")
                answer = "I'm currently experiencing a temporary neural link disruption. Both my primary servers and backup arrays are offline. Let's try again in a moment."


        # 7. Update short-term memory
        self.short_term.add("user", query)
        self.short_term.add("assistant", answer)

        return {
            "answer": answer,
            "query_type": rag_result["query_type"],
            "needs_retrieval": rag_result["needs_retrieval"],
            "chunks_used": len(rag_result["chunks"]),
            "sources": list({c["source"] for c in rag_result["chunks"]}),
            "long_term_memories_used": long_term_memories,
        }

    def _build_prompt(
        self,
        query: str,
        context: str,
        analogy: str | None,
        conversation_history: str,
        long_term_prompt_block: str,
        system_prompt: str,
        query_type: str,
    ) -> str:
        parts = [system_prompt, "\n\n"]

        if long_term_prompt_block:
            parts.append(f"{long_term_prompt_block}\n\n")

        if conversation_history:
            parts.append(f"CONVERSATION SO FAR:\n{conversation_history}\n\n")

        if context:
            parts.append(f"RELEVANT CONTEXT FROM YOUR ACTUAL WORDS AND INTERVIEWS:\n{context}\n\n")
        elif query_type == "no_retrieval":
            parts.append("(Conversational exchange — respond naturally.)\n\n")
        else:
            parts.append("(No specific source material retrieved — draw from your general knowledge as Jensen.)\n\n")

        if analogy:
            parts.append(f"CONSIDER USING THIS ANALOGY IF RELEVANT:\n{analogy}\n\n")

        parts.append(f"User: {query}\nJensen:")
        return "".join(parts)

    def end_session(self):
        """Extract memories from this session and persist to SQLite."""
        history = self.short_term.get_history()
        if len(history) >= 4:
            save_session_memories(self.user_id, history)
        self.short_term.clear()

    def reset_conversation(self):
        self.short_term.clear()


if __name__ == "__main__":
    agent = JensenAgent(user_id="test_user")
    tests = [
        "Hey thanks, that was really interesting!",
        "What is CUDA and why did you build it?",
        "How did the AlexNet moment change NVIDIA's roadmap?",
    ]
    for q in tests:
        print(f"\nUser: {q}")
        result = agent.chat(q)
        print(f"Jensen: {result['answer'][:300]}...")
        print(f"[type={result['query_type']} retrieval={result['needs_retrieval']} chunks={result['chunks_used']}]")
    agent.end_session()
