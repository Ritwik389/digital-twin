import os
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
import uvicorn

from dotenv import load_dotenv
load_dotenv(".env.local")
# Import existing core modules
from agent import JensenAgent
from memory import get_all_memories, delete_all_memories
import tts as tts_module
from ingest import ingest_all

# os.environ["HF_HOME"] = "./model_cache"
# os.environ["TORCH_HOME"] = "./model_cache"

os.environ["OPENAI_API_BASE"] = "https://api.groq.com/openai/v1"
os.environ["OPENAI_API_KEY"] = os.environ.get("GROQ_API_KEY", "your-key-if-not-in-env")



app = FastAPI(title="Jensen Huang - Digital Twin API")



# Mount the static directory for HTML, CSS, JS, and Images
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize ChromaDB on startup
if not os.path.exists("./storage/chroma_db"):
    print("Creating ChromaDB...")
    ingest_all()
else:
    print("ChromaDB already exists. Skipping.")

# State management for agents (similar to Gradio's state)
agent_instances = {}

def get_agent(user_id: str) -> JensenAgent:
    user_id = user_id.strip() or "default_user"
    
    if len(agent_instances) > 10:
        oldest_key = next(iter(agent_instances))
        del agent_instances[oldest_key]
        
    if user_id not in agent_instances:
        agent_instances[user_id] = JensenAgent(user_id=user_id)
    return agent_instances[user_id]

# --- API Models ---
class ChatRequest(BaseModel):
    message: str
    user_id: str = "default_user"
    era: str = "all"
    voice_enabled: bool = False
    api_key: str = "" 

class SessionRequest(BaseModel):
    user_id: str = "default_user"

# ADD THIS NEW MODEL FOR THE GREETING
class GreetRequest(BaseModel):
    name: str = "Guest"


# --- Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()
@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    if not req.message or not req.message.strip():
        raise HTTPException(status_code=400, detail="Message is required")
    if req.api_key:
        os.environ["GEMINI_API_KEY"] = req.api_key

    agent = get_agent(req.user_id)
    agent.set_era(req.era)

    # Directly query the agent (The prompt guardrail handles the filtering natively)
    result = agent.chat(req.message)
    
    answer = result["answer"]
    sources = result["sources"]
    query_type = result.get("query_type", "unknown")
    chunks_used = result.get("chunks_used", 0)
    
    # Audio Generation
    audio_filename = None
    tts_status = "disabled"
    if req.voice_enabled and tts_module.is_available():
        audio_path = tts_module.speak(answer)
        if audio_path:
            audio_filename = os.path.basename(audio_path)
            tts_status = "success"
        else:
            tts_status = f"error: {tts_module.get_error()}"

    return {
        "answer": answer,
        "sources": sources,
        "query_type": query_type,
        "chunks_used": chunks_used,
        "audio_file": audio_filename,
        "tts_status": tts_status
    }
@app.get("/api/audio/{filename}")
async def get_audio(filename: str):
    """Serves the generated TTS audio files."""
    # Assuming tts.py saves audio in the root directory. Update path if needed.
    file_path = os.path.join(".", filename) 
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="audio/wav")
    raise HTTPException(status_code=404, detail="Audio file not found")

@app.post("/api/greet")
async def get_greeting(req: GreetRequest):
    # Uses the name from the landing page!
    greeting_text = f"Ah, {req.name}, right on time. Drop your bag, and take a seat. Welcome. How are you?"
    
    audio_filename = None
    if tts_module.is_available():
        audio_path = tts_module.speak(greeting_text)
        if audio_path:
            audio_filename = os.path.basename(audio_path)
            
    return {
        "text": greeting_text,
        "audio_file": audio_filename
    }


@app.post("/api/session/save")
async def save_session(req: SessionRequest):
    agent = get_agent(req.user_id)
    agent.end_session()
    return {"status": "Session saved — facts extracted into long-term memory."}

@app.post("/api/session/reset")
async def reset_session(req: SessionRequest):
    agent = get_agent(req.user_id)
    agent.reset_conversation()
    return {"status": "Chat reset."}

@app.get("/api/memory/{user_id}")
async def fetch_memories(user_id: str):
    facts = get_all_memories(user_id.strip() or "default_user")
    return {"memories": facts}

@app.delete("/api/memory/{user_id}")
async def clear_memories(user_id: str):
    delete_all_memories(user_id.strip() or "default_user")
    return {"status": "Memories cleared."}

if __name__ == "__main__":
    uvicorn.run(app, port=8000)