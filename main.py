"""FastAPI chat, isolated audio, and Neo4j memory routes."""
from collections import OrderedDict
import asyncio
from contextlib import asynccontextmanager
import logging
import re
import threading
import hashlib
from typing import Literal

from dotenv import load_dotenv
load_dotenv('.env.local')

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field
import uvicorn

from agent import JensenAgent
from api_client import GenerationError, GenerationCancelled, _collect_keys
from knowledge_graph import graph, UPDATE_EVERY
from knowledge_map import knowledge_map
from memory import get_all_memories
import tts as tts_module
from streaming import event_response
from source_utils import in_era

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app):
    def prepare_map():
        knowledge_map.initialize()
        if _collect_keys() or __import__('os').getenv('GROQ_API_KEY'): knowledge_map.enrich()
    task = asyncio.create_task(asyncio.to_thread(prepare_map))
    try:
        yield
    finally:
        await task
        graph.close()


app = FastAPI(title='Jensen Huang - Digital Twin API', lifespan=lifespan)
app.mount('/static', StaticFiles(directory='static'), name='static')
agent_instances = OrderedDict()
_agents_lock = threading.RLock()
_conversation_locks = [threading.RLock() for _ in range(128)]


def conversation_lock(user_id):
    """Stable across cache eviction; also serializes reset/save/clear with turns."""
    index = int(hashlib.sha256(user_id.encode()).hexdigest(), 16) % len(_conversation_locks)
    return _conversation_locks[index]


def get_agent(user_id, session_id='default'):
    key = (user_id, session_id)
    with _agents_lock:
        if key not in agent_instances:
            agent_instances[key] = JensenAgent(user_id=user_id, session_id=session_id)
        agent_instances.move_to_end(key)
        while len(agent_instances) > 100:
            agent_instances.popitem(last=False)
        return agent_instances[key]


class SessionRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=128)
    session_id: str = Field(default='default', min_length=1, max_length=128)


class ChatRequest(SessionRequest):
    message: str = Field(min_length=1, max_length=8000)
    era: Literal['all', 'pre_cuda', 'deep_learning', 'llm_era'] = 'all'
    voice_enabled: bool = False
    knowledge_node_id: str | None = Field(default=None, max_length=80)
    mode: Literal['conversation','beginner','pitch','architecture','compare'] = 'conversation'
    compare_years: list[int] | None = Field(default=None, min_length=2, max_length=2)
    use_graph: bool = True
    retrieval_strategy: Literal['vector', 'reranked'] = 'reranked'


def validate_comparison(req):
    if req.compare_years and (len(set(req.compare_years)) != 2 or any(y < 1900 or y > 2100 or not in_era(y,req.era) for y in req.compare_years)):
        raise HTTPException(400,'Choose two different source years within the selected era, or select All Eras.')


class GreetRequest(BaseModel):
    name: str = Field(default='Guest', max_length=80)
    voice_enabled: bool = False


class SpeakRequest(BaseModel):
    text: str = Field(min_length=1, max_length=6000)


@app.get('/', response_class=HTMLResponse)
def serve_frontend():
    with open('static/index.html', encoding='utf-8') as f:
        return f.read()


@app.get('/api/health')
def health():
    import os
    return {'status': 'ok', 'chat_configured': bool(_collect_keys() or os.getenv('GROQ_API_KEY')),
            'tts_backend': 'pocket', 'tts_voice': 'jensen_reference', 'tts_loaded': tts_module.is_loaded(),
            'neo4j_configured': graph.configured(), 'memory_update_turns': UPDATE_EVERY,
            'chat_status': 'configured' if _collect_keys() or os.getenv('GROQ_API_KEY') else 'not_configured',
            'voice_status': 'ready' if tts_module.is_loaded() else ('unavailable' if tts_module.get_error() else 'not_initialized'),
            'voice_error': tts_module.get_error(), 'session_ownership': 'demo_visitor_id'}


@app.post('/api/chat')
def chat_endpoint(req: ChatRequest, background_tasks: BackgroundTasks):
    if not req.message.strip():
        raise HTTPException(400, 'Message is required')
    validate_comparison(req)
    knowledge_node = None
    if req.knowledge_node_id:
        knowledge_node = knowledge_map.node(req.knowledge_node_id)
        if knowledge_node is None:
            raise HTTPException(404, 'This source topic is unavailable. Refresh the Jensen map.')
    try:
        with conversation_lock(req.user_id):
            agent = get_agent(req.user_id, req.session_id)
            with agent.lock:
                agent.set_era(req.era)
                result = agent.chat(req.message.strip(), knowledge_node=knowledge_node, mode=req.mode, compare_years=req.compare_years, use_graph=req.use_graph, retrieval_strategy=req.retrieval_strategy)
    except GenerationError as exc:
        raise HTTPException(503, str(exc)) from None
    except Exception:
        logger.exception('Chat request failed')
        raise HTTPException(500, 'Could not generate a reply. Please try again.') from None
    background_tasks.add_task(graph.update, req.user_id)
    # Voice is generated separately by the browser so it cannot delay a text answer.
    return result | {'audio_file': None, 'tts_status': ('unavailable' if tts_module.get_error() else 'pending') if req.voice_enabled else 'disabled',
                     'memory': graph.status(req.user_id)}


@app.post('/api/chat/stream')
def chat_stream(req: ChatRequest):
    if not req.message.strip(): raise HTTPException(400,'Message is required')
    node = knowledge_map.node(req.knowledge_node_id) if req.knowledge_node_id else None
    if req.knowledge_node_id and node is None: raise HTTPException(404,'Source topic unavailable')
    validate_comparison(req)
    def worker(emit,cancel):
        acquired=False
        lock=conversation_lock(req.user_id)
        try:
            while not cancel.is_set():
                if lock.acquire(timeout=.2): acquired=True;break
            if not acquired:return
            # Lookup only after acquiring the stable lock; a waiting request
            # must not keep an agent made stale by eviction or memory clearing.
            agent = get_agent(req.user_id,req.session_id)
            agent.set_era(req.era)
            for event in agent.stream(req.message.strip(),node,req.mode,req.compare_years,cancel,req.use_graph,req.retrieval_strategy): emit(event)
            if not cancel.is_set(): threading.Thread(target=graph.update,args=(req.user_id,),daemon=True).start()
        except GenerationCancelled: pass
        except GenerationError as exc: emit({'type':'error','error':str(exc),'code':'generation_unavailable','retryable':True})
        except Exception:
            logger.exception('Chat stream failed');emit({'type':'error','error':'Could not complete the reply. Please retry.'})
        finally:
            if acquired:lock.release()
    return event_response(worker)


class StreamSpeakRequest(SpeakRequest):
    request_id: str = Field(pattern=r'^[a-f0-9-]{36}$')

class CancelSpeakRequest(BaseModel):
    request_id: str = Field(pattern=r'^[a-f0-9-]{36}$')

_voice_jobs={}
_voice_jobs_lock=threading.Lock()

@app.post('/api/speak/stream')
def stream_speech(req: StreamSpeakRequest):
    key=req.request_id
    if not req.text.strip():raise HTTPException(400,'Text is required')
    cancel=threading.Event()
    with _voice_jobs_lock:
        if key in _voice_jobs:raise HTTPException(409,'Voice request is already active')
        if len(_voice_jobs)>=8:raise HTTPException(429,'Voice queue is full. Retry shortly.')
        _voice_jobs[key]=cancel
    def worker(emit,stop):
        try:
            for event in tts_module.speak_stream(req.text,stop):emit(event)
        except tts_module.VoiceSetupError as exc:emit({'type':'error','error':str(exc)+' You can still read the reply.','code':'tts_unavailable','retryable':True})
        except Exception:
            logger.exception('Voice stream failed');emit({'type':'error','error':'Voice could not finish. You can still read the reply or retry Play.','code':'tts_unavailable','retryable':True})
        finally:
            with _voice_jobs_lock:_voice_jobs.pop(key,None)
    return event_response(worker,cancel)

@app.post('/api/speak/cancel')
def cancel_speech(req: CancelSpeakRequest):
    key=req.request_id
    with _voice_jobs_lock:
        job=_voice_jobs.get(key)
        if job:job.set()
    return {'status':'stopped'}


@app.post('/api/speak')
def generate_speech(req: SpeakRequest):
    if not req.text.strip():
        raise HTTPException(400, 'Text is required')
    audio_path = tts_module.speak(req.text)
    if not audio_path:
        raise HTTPException(503, tts_module.get_error() or 'Voice is temporarily unavailable.')
    from pathlib import Path
    return {'audio_file': Path(audio_path).name, 'tts_backend': 'pocket', 'tts_voice': 'jensen_reference'}


@app.get('/api/audio/{filename}')
def get_audio(filename: str):
    if not re.fullmatch(r'[a-f0-9]{32}\.wav', filename):
        raise HTTPException(404, 'Audio file not found')
    path = tts_module.AUDIO_DIR / filename
    if not path.is_file():
        raise HTTPException(404, 'Audio expired. Click Play to regenerate it.')
    return FileResponse(path, media_type='audio/wav', headers={'Cache-Control': 'private, max-age=3600'})


@app.post('/api/greet')
def get_greeting(req: GreetRequest):
    return {'text': f'Ah, {req.name or "Guest"}, right on time. Drop your bag, and take a seat. Welcome. How are you?',
            'audio_file': None, 'tts_status': 'pending' if req.voice_enabled else 'disabled'}


@app.post('/api/session/save')
def save_session(req: SessionRequest):
    with conversation_lock(req.user_id):
        agent = get_agent(req.user_id, req.session_id)
        with agent.lock:
            state = agent.end_session()
    if state.get('state') == 'error':
        raise HTTPException(503, state.get('error'))
    return {'status': 'Memory saved.', 'memory': state}


@app.post('/api/session/reset')
def reset_session(req: SessionRequest):
    with conversation_lock(req.user_id):
        agent = get_agent(req.user_id, req.session_id)
        with agent.lock:
            agent.reset_conversation()
    return {'status': 'Chat reset.'}


@app.get('/api/memory/{user_id}')
def fetch_memories(user_id: str):
    return {'memories': graph.facts(user_id)}


@app.get('/api/graph/{user_id}')
def fetch_graph(user_id: str):
    return graph.fetch(user_id)


@app.post('/api/graph/update')
def update_graph(req: SessionRequest):
    return graph.update(req.user_id, force=True)


@app.get('/api/knowledge-map')
def fetch_knowledge_map(background_tasks: BackgroundTasks):
    data = knowledge_map.snapshot()
    if data.get('state') != 'ready' or not data.get('neo4j_synced'):
        background_tasks.add_task(knowledge_map.initialize)
    return data


@app.get('/api/knowledge-map/node/{node_id}')
def fetch_knowledge_node(node_id: str):
    node = knowledge_map.node(node_id)
    if node is None:
        raise HTTPException(404, 'Source topic not found')
    return node


class FactEditRequest(BaseModel):
    user_id: str = Field(min_length=1,max_length=128)
    old_fact: str = Field(min_length=1,max_length=1000)
    fact: str = Field(min_length=1,max_length=1000)
    category: Literal['background','interest','expertise','project','preference','general'] = 'preference'
    importance: int = Field(default=2,ge=1,le=3)

class FactDeleteRequest(BaseModel):
    user_id: str = Field(min_length=1,max_length=128)
    fact: str = Field(min_length=1,max_length=1000)

@app.patch('/api/memory/fact')
def edit_memory(req: FactEditRequest):
    if not req.fact.strip():raise HTTPException(400,'A remembered fact cannot be blank')
    try:return {'memories':graph.edit_fact(req.user_id,req.old_fact,req.fact.strip(),req.category,req.importance)}
    except Exception:
        logger.exception('Memory correction failed');raise HTTPException(503,'Could not correct this memory. Retry shortly.') from None

@app.delete('/api/memory/fact')
def forget_memory(req: FactDeleteRequest):
    try:return {'memories':graph.edit_fact(req.user_id,req.fact)}
    except Exception:
        logger.exception('Memory deletion failed');raise HTTPException(503,'Could not forget this memory. Retry shortly.') from None

class RetrievalRequest(BaseModel):
    query: str = Field(min_length=1,max_length=2000)
    era: Literal['all','pre_cuda','deep_learning','llm_era'] = 'all'
    strategy: Literal['vector','reranked'] = 'reranked'
    strict_reranker: bool = False

@app.post('/api/retrieval/inspect')
def inspect_retrieval(req: RetrievalRequest):
    import time
    from rag import retrieve, RerankerUnavailable
    from source_utils import ERA_YEAR_RANGES,citations_for
    start=time.perf_counter();low,high=ERA_YEAR_RANGES[req.era]
    try:result=retrieve(req.query,year_min=low,year_max=high,strategy=req.strategy,strict_reranker=req.strict_reranker)
    except RerankerUnavailable as exc:raise HTTPException(503,str(exc)) from None
    except Exception:
        logger.exception('Retrieval inspection failed')
        raise HTTPException(503,'Source retrieval is temporarily unavailable; metadata filters were preserved.') from None
    return {'query_type':result['query_type'],'citations':citations_for(result['chunks'],req.era),
            'retrieval_ms':round((time.perf_counter()-start)*1000,1),
            **{key:result[key] for key in ('ranking_backend','candidate_ids','candidate_count','candidate_ms','ranking_ms')}}

@app.get('/api/corpus/status')
def corpus_status():
    data=knowledge_map.snapshot()
    return {key:data.get(key) for key in ('state','fingerprint','document_count','source_count','quality','relation_state','semantic_relation_count')}

@app.get('/api/timeline')
def timeline():return {'years':knowledge_map.timeline()}

@app.get('/showcase',response_class=HTMLResponse)
def showcase():return FileResponse('static/showcase.html')

@app.get('/api/benchmark')
def benchmark_results():
    import json
    from pathlib import Path
    path=Path('reports/benchmark.json')
    return json.loads(path.read_text()) if path.exists() else {'state':'not_run','message':'Benchmark results will appear after the evaluation run.'}


@app.get('/api/evaluation')
def paired_evaluation_results():
    import json
    from pathlib import Path
    path=Path('reports/evaluation.json')
    return json.loads(path.read_text()) if path.exists() else {'state':'not_run','message':'The labeled, paired evaluation has not been published yet.'}


@app.delete('/api/memory/{user_id}')
def clear_memories(user_id: str):
    with conversation_lock(user_id):
        return _clear_memories(user_id)


def _clear_memories(user_id):
    try:
        graph.clear(user_id)
    except Exception:
        logger.exception('Memory deletion failed')
        raise HTTPException(503, 'Memory could not be cleared. Please try again.') from None
    with _agents_lock:
        for key in list(agent_instances):
            if key[0] == user_id:
                del agent_instances[key]
    return {'status': 'Memories and knowledge graph cleared.'}


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=7860)
