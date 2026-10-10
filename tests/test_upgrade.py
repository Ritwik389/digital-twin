import base64
import json
import threading
from unittest.mock import Mock

import pytest
import torch
from fastapi.testclient import TestClient

from test_app import isolate, source_documents, configure_graph
import agent
import api_client
import graph_reasoning as gr
import ingest
import knowledge_graph as kg
import knowledge_map as km
import main
import memory
import rag
import source_utils as su
import tts


@pytest.mark.parametrize('query,skip', [('Hi',True),('Hi Jensen!',True),('Thanks!',True),('Hi Jensen, explain CUDA.',False),('Hey, how does NVLink work?',False),('Thank you, explain digital twins.',False)])
def test_only_pure_greetings_skip_retrieval(query,skip):
    assert rag._is_no_retrieval(query)==skip


def test_filtered_retrieval_does_not_retry_without_filter(monkeypatch):
    collection=Mock();collection.count.return_value=10;collection.query.side_effect=RuntimeError('Bad filter')
    monkeypatch.setattr(rag,'get_chroma_collection',lambda:collection)
    with pytest.raises(RuntimeError):rag.retrieve_chunks('Blackwell',year_max=2007)
    assert collection.query.call_count==1
    assert collection.query.call_args.kwargs['where']=={'$and':[{'year':{'$gte':1}},{'year':{'$lte':2007}}]}


def test_follow_up_retrieval_uses_previous_user_topic(monkeypatch):
    retrieve=Mock(return_value=[]);monkeypatch.setattr(rag,'retrieve_chunks',retrieve)
    rag.retrieve('Why is it useful?',conversation_history='User: Explain CUDA\nJensen: Parallel computing')
    assert 'Explain CUDA' in retrieve.call_args.args[0]


def test_citations_filter_eras_and_never_invent_timestamps():
    records=[{'filename':'gtc_2024.txt','year':2024,'text':'A real quote about CUDA','chunk_index':7}, {'year':2007,'text':'A dated old excerpt'}, {'year':0,'text':'Undated'}]
    assert len(su.citations_for(records,'pre_cuda'))==1
    citation=su.citations_for(records)[0]
    assert citation['start_seconds'] is None and 't=' not in (citation['url'] or '')
    result=su.provenance({'filename':'gtc_2024.txt','url':'https://www.youtube.com/watch?v=example','start_seconds':123.5})
    assert 't=123' in result['url']
    assert su.provenance({'url':'javascript:alert(1)'})['url'] is None


def test_semantic_links_require_exact_source_and_both_entities():
    docs=source_documents();snapshot=km.build_map(docs)
    ids={n['label']:n['id'] for n in snapshot['nodes']}
    item={'source_id':ids['Blackwell'],'target_id':ids['NVLink'],'relation':'USES','chunk_id':'three','evidence':'Blackwell uses NVLink for AI factories and inference.'}
    assert gr.validate_relations({'relations':[item]},snapshot['nodes'],docs,gr.aliases(km.CATALOG))[0]['label']=='USES'
    for bad in [item|{'evidence':'Blackwell secretly bought NVLink.'},item|{'relation':'DELETE_ALL'},item|{'evidence':'Robotics can benefit from physical AI.'}]:
        assert not gr.validate_relations({'relations':[bad]},snapshot['nodes'],docs,gr.aliases(km.CATALOG))


def test_graph_paths_are_bounded_and_era_filtered():
    snapshot=km.build_map(source_documents());seed=next(n['id'] for n in snapshot['nodes'] if n['label']=='CUDA')
    paths,backend=gr.retrieve_paths(snapshot,[seed],'llm_era')
    assert paths and backend=='corpus_fallback'
    assert all(len(p['edges'])<=2 for p in paths)
    assert not gr.retrieve_paths(snapshot,[seed],'pre_cuda')[0]


def test_comparison_prompt_cannot_include_other_years(monkeypatch):
    snapshot=km.build_map(source_documents());monkeypatch.setattr(km.knowledge_map,'_snapshot',snapshot)
    monkeypatch.setattr(agent,'retrieve',lambda **k:{'chunks':[],'query_type':'direct'})
    a=agent.JensenAgent('alice');prompt,meta=a.prepare('CUDA',compare_years=[2007,2019])
    assert not meta['citations'] and not meta['graph_paths']
    assert 'SOURCE GRAPH PATHS' not in prompt


class SSE:
    ok=True
    status_code=200
    def __init__(self,lines):self.lines=lines;self.closed=False
    def iter_lines(self,**kwargs):yield from self.lines
    def close(self):self.closed=True


def test_stream_groq_splits_delta_and_done(monkeypatch):
    monkeypatch.setenv('GROQ_API_KEY','test')
    response=SSE(['data: '+json.dumps({'choices':[{'delta':{'content':'Hello'}}]}),'data: '+json.dumps({'choices':[{'delta':{'content':' there'},'finish_reason':'stop'}]}),'data: [DONE]'])
    monkeypatch.setattr(api_client.requests,'post',Mock(return_value=response))
    assert list(api_client.generate_text_stream('hello'))==['Hello',' there']
    assert response.closed


def test_provider_fallback_never_mixes_partial_answers(monkeypatch):
    monkeypatch.setenv('GOOGLE_API_KEY','test');monkeypatch.setenv('GROQ_API_KEY','test')
    response=SSE(['data: '+json.dumps({'candidates':[{'content':{'parts':[{'text':'Partial answer'}]}}]})])
    post=Mock(return_value=response);monkeypatch.setattr(api_client.requests,'post',post)
    stream=api_client.generate_text_stream('hello')
    assert next(stream)=='Partial answer'
    with pytest.raises(api_client.GenerationError):next(stream)
    assert post.call_count==1 and response.closed


def test_cancelled_text_does_not_become_completed_history(monkeypatch):
    monkeypatch.setattr(agent,'retrieve',lambda **k:{'chunks':[],'query_type':'no_retrieval'})
    cancel=threading.Event()
    def pieces(*args):yield 'Partial';cancel.set();yield ' late'
    monkeypatch.setattr(agent,'generate_text_stream',pieces)
    a=agent.JensenAgent('alice');stream=a.stream('Hi',cancel=cancel)
    assert next(stream)['type']=='meta' and next(stream)['text']=='Partial'
    with pytest.raises(api_client.GenerationCancelled):next(stream)
    assert a.short_term.history==[] and kg.restore_history('alice','default')==[]


def test_stream_route_saves_only_completed_turn(monkeypatch):
    monkeypatch.setattr(agent,'retrieve',lambda **k:{'chunks':[],'query_type':'no_retrieval'})
    monkeypatch.setattr(agent,'generate_text_stream',lambda *args:iter(['Hello',' friend']))
    with TestClient(main.app) as client:
        response=client.post('/api/chat/stream',json={'user_id':'alice','message':'Hi'})
        events=[json.loads(line) for line in response.text.splitlines()]
        assert [e['type'] for e in events]==['meta','delta','delta','done']
        assert events[-1]['answer']=='Hello friend' and events[-1]['first_token_ms']>=0
        assert len(kg.restore_history('alice','default'))==2
        assert client.post('/api/chat',json={'user_id':'alice','message':'Compare','compare_years':[2024,2025],'era':'pre_cuda'}).status_code==400


def test_pocket_stream_cancel_stops_model_and_releases_slot(monkeypatch):
    stop=threading.Event();received=[]
    def generate(state,text,**kwargs):
        received.append(kwargs);yield torch.ones(100)*.2
        if not kwargs['stop'].is_set():yield torch.ones(100)*.2
    monkeypatch.setattr(tts,'_model',Mock(generate_audio_stream=generate,sample_rate=24000))
    monkeypatch.setattr(tts,'_voice_state',{'clone':'jensen'})
    stream=tts.speak_stream('Hello [1]',stop)
    assert next(stream)['type']=='status' and next(stream)['type']=='start'
    event=next(stream);assert len(base64.b64decode(event['pcm']))==200 and event['sample_rate']==24000
    stop.set();assert list(stream)==[]
    assert received[0]['stop'] is stop and received[0]['copy_state'] is True
    assert not list(tts.AUDIO_DIR.glob('*.wav'))
    assert tts._voice_slots.acquire(blocking=False);tts._voice_slots.release()


def test_fact_edit_and_forget_remain_visitor_scoped():
    for visitor in ['alice','bob']:memory.LongTermMemory(visitor).save_facts([{'fact':'I prefer Python','category':'preference','importance':2}])
    facts=kg.graph.edit_fact('alice','I prefer Python','I prefer Rust')
    assert [f['fact'] for f in facts]==['I prefer Rust']
    memory.LongTermMemory('alice').save_facts([{'fact':'I prefer Python'}])
    assert [f['fact'] for f in memory.get_all_memories('alice')]==['I prefer Rust']
    assert memory.get_all_memories('bob')[0]['fact']=='I prefer Python'
    assert kg.graph.edit_fact('alice','I prefer Rust')==[]
    assert kg.graph.edit_fact('alice','I prefer Python','I prefer Rust')[0]['fact']=='I prefer Rust'


def test_ingestion_replaces_changed_chunks_only_after_success(tmp_path):
    path=tmp_path/'talk.txt';path.write_text('year: 2024\n---\nNew CUDA source content')
    collection=Mock();collection.get.return_value={'ids':['old-0'],'metadatas':[{'corpus_version':'old'}]}
    chunker=Mock();chunker.chunk.return_value=['New CUDA source content']
    ingest.ingest_file(path,collection,chunker)
    collection.upsert.assert_called_once();collection.delete.assert_called_once_with(ids=['old-0'])
    collection.reset_mock();collection.upsert.side_effect=RuntimeError('Failed')
    with pytest.raises(RuntimeError):ingest.ingest_file(path,collection,chunker)
    collection.delete.assert_not_called()


def test_changed_corpus_changes_fingerprint_and_preserves_quality_flags():
    docs=source_documents();before=km.build_map(docs)
    after=km.build_map(docs+[docs[0]|{'id':'duplicate'}])
    assert before['fingerprint']!=after['fingerprint']
    assert after['quality']['duplicate_passages']==1


def test_misdated_biography_cannot_enter_historical_answers():
    record={'filename':'jensen_personal.txt','year':2007,'text':'In 2026, Huang was appointed to a council.'}
    assert su.provenance(record)['year']==0
    assert not su.citations_for([record],'pre_cuda')
    assert su.citations_for([record])[0]['date_note']


def test_verified_milken_date_is_repaired_without_touching_text():
    docs=[{'id':'milken','text':'Original source text','metadata':{'filename':'milken_2026.txt','year':2026}}]
    collection=Mock();su.repair_source_dates(collection,docs)
    assert docs[0]['text']=='Original source text'
    assert docs[0]['metadata']['year']==2025 and docs[0]['metadata']['declared_year']==2026
    collection.update.assert_called_once()


def test_grouped_citations_become_clickable_and_graph_ids_stay_out():
    assert su.normalize_citations('Supported [1, 2, 99]. [3, CO_MENTIONED:map_deadbeef]',{1,2,3})=='Supported [1][2]. [3]'


def test_source_prompt_is_bounded_and_uses_readable_graph_labels(monkeypatch):
    docs=source_documents();snapshot=km.build_map(docs)
    monkeypatch.setattr(km.knowledge_map,'_snapshot',snapshot)
    monkeypatch.setattr(agent,'retrieve',lambda **k:{'chunks':[{'text':'CUDA is useful. '*1000,'source':'talk','filename':'talk.txt','year':2025,'chunk_index':i} for i in range(5)],'query_type':'direct'})
    a=agent.JensenAgent('alice');a.short_term.history=[{'role':'user','content':'Prior context '*1000},{'role':'assistant','content':'Prior reply '*1000}]*8
    prompt,meta=a.prepare('Explain CUDA')
    assert len(prompt)<16000 and 'map_' not in prompt
    assert all(len(c['quote'])<=500 for c in meta['citations'])


def test_missing_era_coverage_returns_without_inventing_history(monkeypatch):
    monkeypatch.setattr(agent,'retrieve',lambda **k:{'chunks':[],'query_type':'no_context'})
    generate=Mock(side_effect=AssertionError('No model needed for missing dated sources'));monkeypatch.setattr(agent,'generate_text',generate)
    a=agent.JensenAgent('alice');a.set_era('pre_cuda');result=a.chat('Blackwell in 2007?')
    assert 'reliable source material' in result['answer'] and not result['citations']
    generate.assert_not_called()


def test_long_active_question_keeps_prompt_within_provider_budget(monkeypatch):
    monkeypatch.setattr(agent,'retrieve',lambda **k:{'chunks':[{'text':'CUDA source text. '*1000,'filename':'talk.txt','year':2025,'chunk_index':i} for i in range(5)],'query_type':'direct'})
    monkeypatch.setattr(agent,'get_relevant_memories',lambda *args:['Remembered context '*100]*6)
    prompt,_=agent.JensenAgent('alice').prepare('CUDA '*1500)
    assert len(prompt)<16000 and ('CUDA '*1500).strip() in prompt


def test_relationship_checks_reject_co_mentions_and_invalid_event_types():
    docs=source_documents()+[{'id':'weak','text':'GeForce brought CUDA to the world. CUDA enabled AI.','metadata':{'year':2025}}]
    snapshot=km.build_map(docs);entities={n['label']:n for n in snapshot['nodes'] if n['kind']!='Category'};patterns=gr.aliases(km.CATALOG)
    assert not gr.grounded_predicate('ENABLES',entities['GeForce'],entities['Artificial intelligence'],docs[-1]['text'],patterns)
    assert not gr.grounded_predicate('DISCUSSED_AT',entities['NVIDIA'],entities['Omniverse'],'NVIDIA and Omniverse intersect at a virtual world.',patterns)
    assert not gr.grounded_predicate('PARTNERED_WITH',entities['NVIDIA'],{'label':'Microsoft','kind':'Organization'},'NVIDIA and Microsoft are going to reinvent the PC.',patterns)
    assert gr.grounded_predicate('ENABLES',entities['Physical AI'],entities['Robotics'],'Robotics has been enabled by physical AI.',patterns)


def test_validator_upgrade_reuses_same_corpus_and_drops_weak_cached_links(monkeypatch):
    docs=source_documents();snapshot=km.build_map(docs);nodes={n['label']:n for n in snapshot['nodes']}
    good={'source':nodes['Blackwell']['id'],'target':nodes['NVLink']['id'],'label':'USES','weight':1,'evidence':[su.provenance(docs[2]['metadata'] | {'chunk_id':'three','quote':'Blackwell uses NVLink for AI factories and inference.'})],'support':'exact_source_excerpt'}
    previous=snapshot | {'fingerprint':km.source_fingerprint(docs,'source-map-v2'),'edges':snapshot['edges']+[good,good | {'label':'INTRODUCED'}]}
    configure_graph(monkeypatch);query=Mock(side_effect=[[{'relations':None,'previous':json.dumps(previous)}],[],[],[]]);monkeypatch.setattr(kg.graph,'_query',query)
    instance=km.KnowledgeMap();instance._snapshot=snapshot;instance._documents=docs
    instance.enrich()
    assert instance.snapshot()['relation_state']=='ready' and instance.snapshot()['semantic_relation_count']==1
    assert instance.snapshot()['edges'][-1]['label']=='USES'


def test_cached_links_are_reactivated_after_startup_base_sync(monkeypatch):
    docs=source_documents();snapshot=km.build_map(docs);nodes={n['label']:n for n in snapshot['nodes']}
    good={'source':nodes['Blackwell']['id'],'target':nodes['NVLink']['id'],'label':'USES','weight':1,'evidence':[su.provenance(docs[2]['metadata'] | {'chunk_id':'three','quote':'Blackwell uses NVLink for AI factories and inference.'})],'support':'exact_source_excerpt'}
    configure_graph(monkeypatch);query=Mock(side_effect=[[{'relations':json.dumps([good])}],[],[],[]]);monkeypatch.setattr(kg.graph,'_query',query)
    instance=km.KnowledgeMap();instance._snapshot=snapshot;instance._documents=docs
    instance.enrich()
    assert instance.snapshot()['semantic_relation_count']==1
    sync=next(call.args[0] for call in query.call_args_list if 'UNWIND $nodes' in call.args[0])
    assert 'previous_relations_json' in sync


def test_stream_rate_limit_wait_uses_provider_delay_and_is_cancellable(monkeypatch):
    limited=Mock(ok=False,status_code=429,headers={'retry-after':'7'})
    success=SSE(['data: '+json.dumps({'choices':[{'delta':{'content':'Recovered'},'finish_reason':'stop'}]}),'data: [DONE]'])
    monkeypatch.setattr(api_client.requests,'post',Mock(side_effect=[limited,success]))
    cancel=Mock();cancel.is_set.return_value=False;cancel.wait.return_value=False
    assert list(api_client._sse_stream('https://example.test',{}, {},'groq',cancel))==['Recovered']
    cancel.wait.assert_called_once_with(7)
    monkeypatch.setattr(api_client.requests,'post',Mock(return_value=limited));cancel.wait.return_value=True
    with pytest.raises(api_client.GenerationCancelled):list(api_client._sse_stream('https://example.test',{}, {},'groq',cancel))
