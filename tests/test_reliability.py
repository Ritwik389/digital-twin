from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
from unittest.mock import Mock

from fastapi.testclient import TestClient
import pytest

from test_app import isolate
import agent
import knowledge_graph as kg
import main
import rag
import tts
from evaluation_metrics import rank_metrics, review_metrics, sample_digest


def test_eviction_cannot_split_concurrent_updates_and_health_stays_responsive(monkeypatch):
    started=threading.Event();release=threading.Event();prompts=[]
    monkeypatch.setattr(agent,'retrieve',lambda **k:{'chunks':[],'query_type':'no_retrieval'})
    def generate(prompt):
        prompts.append(prompt)
        if prompt.endswith('User: first\nJensen:'):
            started.set()
            assert release.wait(5)
        return 'Complete reply'
    monkeypatch.setattr(agent,'generate_text',generate)
    with TestClient(main.app) as one, TestClient(main.app) as two, ThreadPoolExecutor(max_workers=3) as pool:
        first=pool.submit(one.post,'/api/chat',json={'user_id':'alice','session_id':'a','message':'first'})
        assert started.wait(3)
        # Exercise the actual problem: evict while the first agent is in use.
        with main._agents_lock:main.agent_instances.clear()
        second=pool.submit(two.post,'/api/chat',json={'user_id':'alice','session_id':'a','message':'second'})
        health=pool.submit(two.get,'/api/health')
        try:
            assert health.result(timeout=2).status_code==200
        finally:release.set()
        assert first.result(timeout=5).status_code==200
        assert second.result(timeout=5).status_code==200
    history=kg.restore_history('alice','a')
    assert [t['content'] for t in history if t['role']=='user']==['first','second']
    assert 'user: first' in prompts[-1] and 'assistant: Complete reply' in prompts[-1]


def test_returned_metadata_is_defensively_filtered(monkeypatch):
    collection=Mock();collection.count.return_value=3
    collection.query.return_value={'ids':[['new','wrong-domain','valid']],
        'documents':[['future','wrong domain','valid']],
        'metadatas':[[{'year':2025,'domain':'gpu'},{'year':2019,'domain':'other'},{'year':2019,'domain':'gpu'}]],
        'distances':[[.1,.2,.3]]}
    monkeypatch.setattr(rag,'get_chroma_collection',lambda:collection)
    assert [c['chunk_id'] for c in rag.retrieve_chunks('GPU',year_max=2021,domain_filter='gpu')]==['valid']
    assert collection.query.call_count==1


def test_reset_waits_for_active_stream_even_after_eviction(monkeypatch):
    started=threading.Event();release=threading.Event();reset_entered=threading.Event()
    original_lock=main.conversation_lock
    def lock(user_id):
        if started.is_set():reset_entered.set()
        return original_lock(user_id)
    monkeypatch.setattr(main,'conversation_lock',lock)
    monkeypatch.setattr(agent,'retrieve',lambda **k:{'chunks':[],'query_type':'no_retrieval'})
    def generate(prompt,cancel):
        started.set()
        assert release.wait(5)
        yield 'Completed streamed reply'
    monkeypatch.setattr(agent,'generate_text_stream',generate)
    with TestClient(main.app) as one, TestClient(main.app) as two, ThreadPoolExecutor(max_workers=2) as pool:
        stream=pool.submit(one.post,'/api/chat/stream',json={'user_id':'alice','session_id':'s','message':'Hello'})
        assert started.wait(3)
        with main._agents_lock:main.agent_instances.clear()
        reset=pool.submit(two.post,'/api/session/reset',json={'user_id':'alice','session_id':'s'})
        try:assert reset_entered.wait(3)
        finally:release.set()
        events=[json.loads(line) for line in stream.result(timeout=5).text.splitlines()]
        assert events[-1]['type']=='done'
        assert reset.result(timeout=5).status_code==200
    assert kg.restore_history('alice','s')==[]


def test_reranker_failure_is_visible_and_strict_evaluation_refuses_it(monkeypatch):
    chunk={'chunk_id':'one','text':'GPU source','distance':.2,'source':'talk','year':2025}
    monkeypatch.setattr(rag,'retrieve_chunks',lambda *a,**k:[dict(chunk)])
    monkeypatch.setattr(rag,'_get_reranker',lambda:None)
    assert rag.retrieve('GPU')['ranking_backend']=='vector_fallback'
    assert rag.retrieve('GPU',strategy='vector')['ranking_backend']=='vector'
    with pytest.raises(rag.RerankerUnavailable):rag.retrieve('GPU',strict_reranker=True)
    monkeypatch.setattr(rag,'_get_reranker',lambda:Mock(predict=Mock(return_value=[float('nan')])))
    with pytest.raises(rag.RerankerUnavailable):rag.retrieve('GPU',strict_reranker=True)


def test_voice_failure_keeps_text_usable_and_has_explicit_error(monkeypatch):
    def unavailable(text,cancel):
        raise tts.VoiceSetupError('Voice is unavailable.')
        yield
    monkeypatch.setattr(tts,'speak_stream',unavailable)
    with TestClient(main.app) as client:
        response=client.post('/api/speak/stream',json={'text':'Readable reply','request_id':'12345678-1234-1234-1234-123456789012'})
        events=[json.loads(line) for line in response.text.splitlines()]
        assert events[-1]['code']=='tts_unavailable'
        assert 'still read the reply' in events[-1]['error']


def test_metrics_distinguish_absence_from_positive_passage_recall():
    case={'relevant_passages':[{'chunk_id':'a','grade':2},{'chunk_id':'b','grade':2}]}
    scores=rank_metrics(case,['x','a','a','b'])
    assert scores['recall']==1 and scores['mrr']==.5 and 0<scores['ndcg']<1
    assert rank_metrics({'relevant_passages':[]},[])['recall'] is None
    assert rank_metrics(case,[])=={'recall':0,'mrr':0,'ndcg':0}


def test_manual_scores_require_matching_response_and_do_not_count_errors():
    digest=sample_digest('I have no evidence.',[])
    sample={'question_id':'q1','strategy':'vector','completed':True,'expected_behavior':'abstain','response_digest':digest}
    review={'question_id':'q1','strategy':'vector','response_digest':digest,'abstained':True,'claims':[]}
    assert review_metrics([sample],[review])['abstention_recall']==1
    assert review_metrics([sample],[review|{'response_digest':'stale'}])['reviewed_responses']==0
    assert review_metrics([sample|{'completed':False}],[review])['reviewed_responses']==0


def test_container_and_compose_use_same_filename_and_port():
    import yaml
    compose=yaml.safe_load(Path('docker-compose.yml').read_text())['services']['jensen-twin']
    assert compose['build']['dockerfile']=='Dockerfile'
    assert compose['ports']==['7860:7860']
    text=Path('Dockerfile').read_text()
    assert 'EXPOSE 7860' in text and '"7860"' in text
    assert 'DockerFile' not in {p.name for p in Path('.').iterdir()}
