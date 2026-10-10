import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import soundfile as sf
import torch
from fastapi.testclient import TestClient

import api_client
import agent
import knowledge_graph as kg
import knowledge_map as km
import main
import memory
import tts


@pytest.fixture(autouse=True)
def isolate(monkeypatch, tmp_path):
    path = tmp_path / 'memory.db'
    monkeypatch.setattr(memory, 'DB_PATH', path)
    monkeypatch.setattr(kg, 'DB_PATH', path)
    for name in ['GOOGLE_API_KEY', 'GEMINI_API_KEY', 'GROQ_API_KEY', 'NEO4J_URI', 'NEO4J_USERNAME', 'NEO4J_PASSWORD', 'NEO4J_DATABASE'] + [f'GOOGLE_API_KEY_{i}' for i in range(1, 4)]:
        monkeypatch.delenv(name, raising=False)
    main.agent_instances.clear()
    kg.graph._errors.clear()
    monkeypatch.setattr(km.knowledge_map, 'initialize', lambda: None)
    monkeypatch.setattr(km.knowledge_map, '_snapshot', None)
    monkeypatch.setattr(km.knowledge_map, '_state', 'building')
    monkeypatch.setattr(km.knowledge_map, '_neo4j_synced', False)
    monkeypatch.setattr(tts, 'AUDIO_DIR', tmp_path / 'audio')
    monkeypatch.setattr(tts, '_model', None)
    monkeypatch.setattr(tts, '_voice_state', None)
    monkeypatch.setattr(tts, '_load_error', None)
    monkeypatch.setattr(tts, '_retry_after', 0)
    yield


def response(status, data):
    return Mock(ok=status < 400, status_code=status, json=lambda: data)


def test_import_and_missing_secrets_do_not_crash():
    with TestClient(main.app) as client:
        assert client.get('/api/health').json()['chat_configured'] is False
        result = client.post('/api/chat', json={'user_id': 'alice', 'message': 'Hello'})
        assert result.status_code == 503
        assert kg.restore_history('alice', 'default') == []


def test_gemini_503_retries_then_current_groq_fallback(monkeypatch):
    monkeypatch.setenv('GOOGLE_API_KEY_1', 'test')
    monkeypatch.setenv('GROQ_API_KEY', 'test')
    post = Mock(side_effect=[response(503, {}), response(503, {}), response(200, {'choices': [{'message': {'content': 'Recovered'}}]})])
    monkeypatch.setattr(api_client.requests, 'post', post)
    monkeypatch.setattr(api_client.time, 'sleep', lambda _: None)
    assert api_client.generate_text('Hello') == 'Recovered'
    assert post.call_args.kwargs['json']['model'] == 'openai/gpt-oss-120b'
    assert post.call_count == 3


def test_groq_only_and_empty_generation(monkeypatch):
    monkeypatch.setenv('GROQ_API_KEY', 'test')
    monkeypatch.setattr(api_client.requests, 'post', lambda *a, **k: response(200, {'choices': [{'message': {'content': ''}}]}))
    with pytest.raises(api_client.GenerationError):
        api_client.generate_text('Hello')


def test_retrieval_failure_still_chats_and_voice_does_not_block(monkeypatch):
    monkeypatch.setattr(agent, 'retrieve', Mock(side_effect=RuntimeError('retrieval offline')))
    monkeypatch.setattr(agent, 'generate_text', lambda p: 'A good reply')
    monkeypatch.setattr(tts, 'speak', Mock(side_effect=AssertionError('TTS must be separate')))
    with TestClient(main.app) as client:
        data = client.post('/api/chat', json={'user_id': 'alice', 'session_id': 'a', 'message': 'CUDA?', 'voice_enabled': True}).json()
        assert data['answer'] == 'A good reply'
        assert data['tts_status'] == 'pending'
        assert data['memory']['pending_turns'] == 1
    assert len(kg.restore_history('alice', 'a')) == 2
    assert not kg.restore_history('alice', 'b')
    assert not kg.restore_history('bob', 'a')


def test_outage_not_saved_as_a_reply(monkeypatch):
    monkeypatch.setattr(agent, 'retrieve', lambda **kwargs: {'context': '', 'chunks': [], 'query_type': 'chat', 'needs_retrieval': False})
    monkeypatch.setattr(agent, 'generate_text', Mock(side_effect=api_client.GenerationError('Busy')))
    with TestClient(main.app) as client:
        assert client.post('/api/chat', json={'user_id': 'alice', 'message': 'Hello'}).status_code == 503
    assert main.get_agent('alice').short_term.get_history() == []
    assert kg.graph.status('alice')['pending_turns'] == 0


def test_audio_is_unique_and_root_files_are_not_served(monkeypatch):
    state = {'jensen': 'encoded reference'}
    synthesize = Mock(return_value=torch.from_numpy(np.sin(np.arange(3200) / 20)))
    monkeypatch.setattr(tts, '_model', Mock(generate_audio=synthesize, sample_rate=32000))
    monkeypatch.setattr(tts, '_voice_state', state)
    with TestClient(main.app) as client:
        one = client.post('/api/speak', json={'text': 'One'}).json()['audio_file']
        two = client.post('/api/speak', json={'text': 'Two'}).json()['audio_file']
        assert one != two
        audio = client.get('/api/audio/' + one)
        assert audio.status_code == 200 and audio.content[:4] == b'RIFF'
        assert sf.info(str(tts.AUDIO_DIR / one)).samplerate == 32000
        assert synthesize.call_args.args == (state, 'Two')
        assert client.get('/api/health').json()['tts_loaded'] is True
        assert client.get('/api/audio/main.py').status_code == 404
        assert client.get('/api/audio/response.wav').status_code == 404
        assert client.post('/api/speak', json={'text': '   '}).status_code == 400


def pocket_loader(monkeypatch, tmp_path, model):
    reference = tmp_path / 'jensen_ref.wav'
    sf.write(str(reference), np.sin(np.arange(2400) / 20), 24000)
    monkeypatch.setattr(tts, 'REFERENCE_AUDIO', reference)
    load = Mock(return_value=model)
    monkeypatch.setitem(sys.modules, 'pocket_tts', SimpleNamespace(TTSModel=SimpleNamespace(load_model=load)))
    return reference, load


def test_clone_initialization_failure_is_not_cached_and_can_retry(monkeypatch, tmp_path):
    model = Mock(has_voice_cloning=True)
    state = {'jensen': 'encoded reference'}
    model.get_state_for_audio_prompt.side_effect = [RuntimeError('encoding failed'), state]
    reference, load = pocket_loader(monkeypatch, tmp_path, model)
    assert tts.is_available() is False
    assert tts._model is None and tts._voice_state is None
    assert tts.is_available() is False
    assert load.call_count == 1  # Cooldown prevents repeated expensive downloads.
    monkeypatch.setattr(tts, '_retry_after', 0)
    assert tts.is_available() is True
    assert tts._voice_state == state
    model.get_state_for_audio_prompt.assert_called_with(reference, truncate=True)
    assert tts.is_available() is True and load.call_count == 2


def test_no_cloning_weights_gives_actionable_error(monkeypatch, tmp_path):
    model = Mock(has_voice_cloning=False)
    pocket_loader(monkeypatch, tmp_path, model)
    monkeypatch.setattr(tts, '_retry_cloning_download', lambda: None)
    with TestClient(main.app) as client:
        result = client.post('/api/speak', json={'text': 'Hello'})
        assert result.status_code == 503 and 'HF_TOKEN' in result.json()['detail']
        assert client.get('/api/health').json()['tts_loaded'] is False
    model.get_state_for_audio_prompt.assert_not_called()


def test_cloning_download_reports_rejection_without_exposing_token(monkeypatch):
    config = SimpleNamespace(weights_path='hf://kyutai/pocket-tts/model.safetensors')
    failure = RuntimeError('sensitive provider details')
    failure.response = SimpleNamespace(status_code=403)
    monkeypatch.setitem(sys.modules, 'pocket_tts.utils.config', SimpleNamespace(CONFIGS_DIR=Path('/configs'), load_config=lambda _: config))
    monkeypatch.setitem(sys.modules, 'pocket_tts.utils.utils', SimpleNamespace(download_if_necessary=Mock(side_effect=failure)))
    with pytest.raises(tts.VoiceSetupError) as caught:
        tts._retry_cloning_download()
    assert 'HTTP 403' in str(caught.value) and 'HF_TOKEN' in str(caught.value)
    assert 'sensitive provider details' not in str(caught.value)


def test_git_lfs_pointer_is_not_treated_as_reference_audio(monkeypatch, tmp_path):
    reference = tmp_path / 'jensen_ref.wav'
    reference.write_text('version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 441044\n')
    monkeypatch.setattr(tts, 'REFERENCE_AUDIO', reference)
    assert tts.is_available() is False
    assert 'Git LFS' in tts.get_error()


def configure_graph(monkeypatch):
    for key in ('NEO4J_URI', 'NEO4J_USERNAME', 'NEO4J_PASSWORD'):
        monkeypatch.setenv(key, 'test')


def test_neo4j_uses_home_database_unless_explicitly_selected(monkeypatch):
    driver = Mock()
    driver.execute_query.return_value.records = []
    monkeypatch.setattr(kg.graph, '_get_driver', lambda: driver)
    kg.graph._query('RETURN 1', read=True)
    assert driver.execute_query.call_args.kwargs['database_'] is None
    assert driver.execute_query.call_args.kwargs['routing_'] == 'r'
    monkeypatch.setenv('NEO4J_DATABASE', 'memory')
    kg.graph._query('RETURN 1')
    assert driver.execute_query.call_args.kwargs['database_'] == 'memory'


def payload():
    return {'facts': [{'fact': 'User is learning CUDA', 'category': 'interest', 'importance': 2}],
            'relations': [{'source': 'Visitor', 'source_type': 'Person', 'relation': 'STUDIES', 'target': 'CUDA', 'target_type': 'Skill', 'evidence': 'I am learning CUDA'}]}


def test_five_and_ten_turn_batches_and_retry(monkeypatch):
    configure_graph(monkeypatch)
    saved = []
    monkeypatch.setattr(kg, 'generate_text', lambda _: json.dumps(payload()))
    monkeypatch.setattr(kg.graph, '_save_batch', lambda user, rows, facts, relations: saved.append((user, [r['id'] for r in rows])))
    for i in range(4): kg.record_turn('alice', 'a', 'I am learning CUDA', 'Okay')
    kg.graph.update('alice')
    assert not saved
    kg.record_turn('alice', 'a', 'I am learning CUDA', 'Okay')
    kg.graph.update('alice')
    assert len(saved) == 1 and len(saved[0][1]) == 5
    for i in range(5): kg.record_turn('alice', 'a', 'I am learning CUDA', 'Okay')
    kg.graph.update('alice')
    assert len(saved) == 2
    assert set(saved[0][1]).isdisjoint(saved[1][1])
    assert kg.graph.status('alice')['pending_turns'] == 0
    assert len(memory.get_all_memories('alice')) == 1
    kg.record_turn('alice', 'a', 'I am learning CUDA', 'Okay')
    monkeypatch.setattr(kg.graph, '_save_batch', Mock(side_effect=RuntimeError('offline')))
    assert kg.graph.update('alice', force=True)['state'] == 'error'
    assert kg.graph.status('alice')['pending_turns'] == 1
    monkeypatch.setattr(kg.graph, '_save_batch', lambda *a: None)
    assert kg.graph.update('alice', force=True)['state'] == 'ready'
    assert kg.graph.status('alice')['pending_turns'] == 0


def test_unconfigured_graph_preserves_pending_turns():
    kg.record_turn('alice', 'a', 'Hi', 'Hello')
    assert kg.graph.update('alice', force=True)['state'] == 'not_configured'
    assert kg.graph.fetch('alice')['nodes'] == []
    assert kg.graph.status('alice')['pending_turns'] == 1


def test_extraction_rejects_cypher_and_unsupported_evidence():
    data = payload()
    data['relations'][0]['relation'] = 'FOO) DETACH DELETE n //'
    facts, relations = kg.validate_extraction(data, [{'user_text': 'I am learning CUDA'}])
    assert not relations
    data = payload(); data['relations'][0]['evidence'] = 'invented quote'
    assert not kg.validate_extraction(data, [{'user_text': 'I am learning CUDA'}])[1]
    assert kg._id('alice', 'Skill', 'CUDA') != kg._id('bob', 'Skill', 'CUDA')


def test_clear_is_visitor_scoped():
    for user in ['alice', 'bob']:
        kg.record_turn(user, 'a', 'Hi', 'Hello')
        memory.LongTermMemory(user).save_facts([{'fact': user, 'category': 'general', 'importance': 1}])
    kg.graph.clear('alice')
    assert not memory.get_all_memories('alice')
    assert kg.graph.status('alice')['pending_turns'] == 0
    assert len(memory.get_all_memories('bob')) == 1
    assert kg.graph.status('bob')['pending_turns'] == 1


def test_reset_history_survives_agent_eviction():
    kg.record_turn('alice', 'a', 'Hello', 'Hi')
    assert len(kg.restore_history('alice', 'a')) == 2
    kg.reset_history('alice', 'a')
    assert kg.restore_history('alice', 'a') == []
    assert kg.graph.status('alice')['pending_turns'] == 1
    kg.record_turn('alice', 'a', 'New', 'New reply')
    assert kg.restore_history('alice', 'a')[0]['content'] == 'New'


def source_documents():
    return [
        {'id': 'one', 'text': 'Jensen Huang explains how NVIDIA CUDA accelerates GPU computing. Omniverse supports digital twins and simulation.',
         'metadata': {'source': 'gtc_keynote', 'filename': 'gtc_2025.txt', 'year': 2025}},
        {'id': 'two', 'text': 'CUDA and NVIDIA enable accelerated computing and deep learning with neural networks.',
         'metadata': {'source': 'interview', 'filename': 'interview_2024.txt', 'year': 2024}},
        {'id': 'three', 'text': 'Blackwell uses NVLink for AI factories and inference. Robotics can benefit from physical AI.',
         'metadata': {'source': 'gtc_keynote', 'filename': 'gtc_2025.txt', 'year': 2025}},
    ]


def test_source_map_uses_actual_passages_and_stable_ids():
    docs = source_documents()
    result = km.build_map(docs)
    entities = {n['label']: n for n in result['nodes'] if n['kind'] != 'Category'}
    assert result['document_count'] == 3 and result['source_count'] == 2
    assert 'CUDA' in entities and 'OpenAI' not in entities
    assert entities['CUDA']['source_count'] == 2
    reversed_cuda = next(n for n in km.build_map(docs[::-1])['nodes'] if n['label'] == 'CUDA')
    assert entities['CUDA']['id'] == reversed_cuda['id']
    for n in entities.values():
        for evidence in n['evidence']:
            assert evidence['quote'] in next(d['text'] for d in docs if d['id'] == evidence['chunk_id'])
    pairs = [e for e in result['edges'] if e['label'] == 'CO_MENTIONED']
    assert pairs
    for edge in pairs:
        assert edge['evidence'][0]['chunk_id'] == edge['evidence'][1]['chunk_id']
    boundary = km.build_map([{'text': 'CUDA compilers compare metadata with nimble, airy code.'}])
    assert {n['label'] for n in boundary['nodes'] if n['kind'] != 'Category'} == {'CUDA', 'Jensen Huang'}


def test_source_node_query_includes_server_selected_evidence_and_is_saved(monkeypatch):
    snapshot = km.build_map(source_documents())
    monkeypatch.setattr(km.knowledge_map, '_snapshot', snapshot)
    node = next(n for n in snapshot['nodes'] if n['label'] == 'CUDA')
    monkeypatch.setattr(agent, 'retrieve', Mock(side_effect=RuntimeError('retrieval offline')))
    generation = Mock(return_value='CUDA accelerates computing.')
    monkeypatch.setattr(agent, 'generate_text', generation)
    with TestClient(main.app) as client:
        public = client.get('/api/knowledge-map').json()
        assert public['nodes'] == snapshot['nodes']
        assert client.get('/api/knowledge-map/node/' + node['id']).json()['evidence'] == node['evidence']
        result = client.post('/api/chat', json={'user_id': 'alice', 'message': node['query'], 'knowledge_node_id': node['id']})
        assert result.status_code == 200
        assert result.json()['sources'] == ['gtc_keynote', 'interview']
        assert node['evidence'][0]['quote'] in generation.call_args.args[0]
        assert 'SOURCE MAP EVIDENCE' in generation.call_args.args[0]
        assert len(kg.restore_history('alice', 'default')) == 2
        assert client.post('/api/chat', json={'user_id': 'alice', 'message': 'Hello', 'knowledge_node_id': 'unknown'}).status_code == 404
        assert generation.call_count == 1
        client.delete('/api/memory/alice')
        assert client.get('/api/knowledge-map').json()['nodes'] == snapshot['nodes']


def test_map_reads_corpus_without_embedding_call_and_mirrors_separate_neo4j_namespace(monkeypatch):
    docs = source_documents()
    collection = Mock()
    collection.count.return_value = len(docs)
    collection.get.return_value = {'ids': [d['id'] for d in docs], 'documents': [d['text'] for d in docs], 'metadatas': [d['metadata'] for d in docs]}
    monkeypatch.setattr(km, 'get_chroma_collection', lambda: collection)
    configure_graph(monkeypatch)
    query = Mock(return_value=[])
    monkeypatch.setattr(kg.graph, '_query', query)
    instance = km.KnowledgeMap()
    instance.initialize()
    assert instance.snapshot()['neo4j_synced'] is True
    assert 'CALL db.labels' in query.call_args_list[0].args[0]
    cypher, parameters = query.call_args.args
    assert 'DTKnowledgeEntity' in cypher and 'MemoryEntity' not in cypher
    assert parameters['map_id'] == km.MAP_ID
    for node in parameters['nodes']:
        assert all(isinstance(value, (str, int)) for value in node.values())
        json.loads(node['evidence_json'])
    instance.initialize()
    assert collection.get.call_count == 1 and query.call_count == 2


def test_map_remains_available_if_neo4j_temporarily_fails(monkeypatch):
    configure_graph(monkeypatch)
    instance = km.KnowledgeMap()
    instance._snapshot = km.build_map(source_documents())
    instance._state = 'ready'
    monkeypatch.setattr(kg.graph, '_query', Mock(side_effect=RuntimeError('offline')))
    instance.initialize()
    assert instance.snapshot()['state'] == 'ready'
    assert instance.snapshot()['neo4j_synced'] is False
    query = Mock(return_value=[])
    monkeypatch.setattr(kg.graph, '_query', query)
    instance.initialize()
    assert instance.snapshot()['neo4j_synced'] is True
