"""Public Jensen source map, built from corpus text and mirrored to Neo4j.

Edges mean corpus co-mentions, not inferred business or personal relationships.
Visitor disclosures are never included in this public dataset.
"""
from collections import Counter, defaultdict
from itertools import combinations
import hashlib
import json
import logging
import re
import threading
import os
from source_utils import effective_year, repair_source_dates, UNVERIFIED_SOURCE_DATES, provenance, in_era
from graph_reasoning import validate_relations, aliases, extract_batch, retrieve_paths, PREDICATES

from chroma_utils import get_chroma_collection
from knowledge_graph import graph

logger = logging.getLogger(__name__)
MAP_ID = 'digital-twin-2-jensen-corpus-v1'
# Curated aliases/taxonomy; an entity enters the map only if corpus text matches.
CATALOG = '''People & partners|Organization|NVIDIA|Nvidia;NVIDIA
People & partners|Person|Jensen Huang|Jensen Huang;Jensen
People & partners|Person|Geoffrey Hinton|Geoffrey Hinton;Hinton
People & partners|Person|Ilya Sutskever|Ilya Sutskever;Sutskever
People & partners|Person|Alex Krizhevsky|Alex Krizhevsky;Krizhevsky
People & partners|Person|Sam Altman|Sam Altman
People & partners|Organization|OpenAI|OpenAI;Open AI
People & partners|Organization|Microsoft|Microsoft
People & partners|Organization|Google|Google
People & partners|Organization|Meta|Meta;Facebook
People & partners|Organization|Amazon|Amazon;AWS
People & partners|Organization|Tesla|Tesla
People & partners|Organization|TSMC|TSMC
People & partners|Organization|Stanford|Stanford
People & partners|Organization|Hugging Face|Hugging Face;HuggingFace
People & partners|Organization|Dell|Dell
People & partners|Organization|Oracle|Oracle
People & partners|Organization|Siemens|Siemens
People & partners|Organization|Cadence|Cadence
People & partners|Organization|Synopsys|Synopsys
People & partners|Organization|Ansys|Ansys
People & partners|Organization|BMW|BMW
People & partners|Organization|Mercedes-Benz|Mercedes;Mercedes-Benz
People & partners|Organization|Foxconn|Foxconn
People & partners|Organization|Disney|Disney
Accelerated computing|Concept|GPU computing|GPU;GPUs;graphics processing
Accelerated computing|Concept|Accelerated computing|accelerated computing;accelerated compute
Accelerated computing|Software|CUDA|CUDA
Accelerated computing|Concept|Parallel computing|parallel computing;parallel processing
Accelerated computing|Concept|Moore’s law|Moore's law;Moore’s law;Moore law
Accelerated computing|Concept|Computer graphics|computer graphics;graphics rendering
Accelerated computing|Concept|Ray tracing|ray tracing;ray-tracing
Accelerated computing|Hardware|GeForce|GeForce
Accelerated computing|Hardware|RTX|RTX
Accelerated computing|Hardware|Tensor Cores|tensor core;tensor cores
Accelerated computing|Software|cuDNN|cuDNN
Accelerated computing|Software|cuBLAS|cuBLAS
Accelerated computing|Software|CUDA-X|CUDA-X;CUDA X
Accelerated computing|Software|RAPIDS|RAPIDS
Accelerated computing|Software|TensorRT|TensorRT
AI & models|Concept|Artificial intelligence|artificial intelligence;AI
AI & models|Concept|Deep learning|deep learning
AI & models|Concept|Neural networks|neural network;neural networks
AI & models|Research|AlexNet|AlexNet;Alex Net
AI & models|Concept|Machine learning|machine learning
AI & models|Concept|Generative AI|generative AI;generative artificial intelligence
AI & models|Concept|Large language models|large language model;large language models;LLM;LLMs
AI & models|Concept|Transformers|transformer;transformers
AI & models|Concept|AI agents|AI agent;AI agents;agentic AI
AI & models|Concept|Inference|inference
AI & models|Concept|Model training|model training;training models;train models
AI & models|Concept|Reinforcement learning|reinforcement learning
AI & models|Concept|Reasoning models|reasoning model;reasoning models
AI & models|Concept|Multimodal AI|multimodal;multi-modal
AI & models|Concept|Synthetic data|synthetic data
AI & models|Product|ChatGPT|ChatGPT;Chat GPT
AI & models|Product|Llama|Llama;LLaMA
AI & models|Product|NeMo|NeMo
AI & models|Product|NIM|NIM;inference microservices
AI & models|Concept|Foundation models|foundation model;foundation models
Chips & infrastructure|Hardware|Blackwell|Blackwell
Chips & infrastructure|Hardware|Hopper|Hopper
Chips & infrastructure|Hardware|Ampere|Ampere
Chips & infrastructure|Hardware|Volta|Volta
Chips & infrastructure|Hardware|Pascal|Pascal
Chips & infrastructure|Hardware|H100|H100
Chips & infrastructure|Hardware|H200|H200
Chips & infrastructure|Hardware|B200|B200
Chips & infrastructure|Hardware|GB200|GB200
Chips & infrastructure|Hardware|GB300|GB300
Chips & infrastructure|Hardware|Grace CPU|Grace CPU;Grace Hopper
Chips & infrastructure|Hardware|DGX|DGX
Chips & infrastructure|Hardware|DGX Spark|DGX Spark
Chips & infrastructure|Hardware|NVLink|NVLink
Chips & infrastructure|Hardware|NVSwitch|NVSwitch
Chips & infrastructure|Hardware|BlueField|BlueField
Chips & infrastructure|Hardware|Spectrum-X|Spectrum-X;Spectrum X
Chips & infrastructure|Hardware|InfiniBand|InfiniBand
Chips & infrastructure|Concept|Data centers|data center;data centers;datacenter
Chips & infrastructure|Concept|AI factories|AI factory;AI factories
Chips & infrastructure|Concept|Supercomputing|supercomputing;supercomputer;supercomputers
Chips & infrastructure|Concept|Cloud computing|cloud computing;cloud infrastructure
Digital & physical worlds|Product|Omniverse|Omniverse
Digital & physical worlds|Concept|Digital twins|digital twin;digital twins
Digital & physical worlds|Concept|Physical AI|physical AI
Digital & physical worlds|Concept|Robotics|robotics;robots;robot
Digital & physical worlds|Product|Isaac|Isaac
Digital & physical worlds|Product|GR00T|GR00T;GROOT
Digital & physical worlds|Product|Cosmos|Cosmos
Digital & physical worlds|Product|NVIDIA DRIVE|NVIDIA DRIVE;Drive platform;Drive Thor
Digital & physical worlds|Concept|Autonomous driving|autonomous driving;autonomous vehicles;self-driving
Digital & physical worlds|Concept|Simulation|simulation;simulations
Digital & physical worlds|Software|OpenUSD|OpenUSD;Universal Scene Description
Digital & physical worlds|Product|Earth-2|Earth-2;Earth 2
Industries & discovery|Industry|Healthcare|healthcare;health care;medical imaging
Industries & discovery|Industry|Drug discovery|drug discovery
Industries & discovery|Product|BioNeMo|BioNeMo
Industries & discovery|Industry|Manufacturing|manufacturing;factories
Industries & discovery|Industry|Scientific computing|scientific computing;scientific discovery
Industries & discovery|Industry|Climate science|climate;weather prediction
Industries & discovery|Industry|Automotive|automotive
Industries & discovery|Industry|Gaming|gaming;video games
Industries & discovery|Industry|Financial services|financial services;finance
Industries & discovery|Industry|Chip design|chip design;design chips;semiconductor design
Industries & discovery|Industry|Logistics|logistics
Industries & discovery|Industry|Telecommunications|telecommunications;telecom;6G
Leadership & ideas|Concept|GTC|GTC;GPU Technology Conference
Leadership & ideas|Concept|Innovation|innovation
Leadership & ideas|Concept|Leadership|leadership
Leadership & ideas|Concept|Resilience|resilience;scar tissue
Leadership & ideas|Concept|First principles|first principles
Leadership & ideas|Concept|Industrial revolution|industrial revolution
Leadership & ideas|Concept|Sovereign AI|sovereign AI
Leadership & ideas|Concept|AI infrastructure|AI infrastructure'''


def node_id(kind, label):
    return 'map_' + hashlib.sha256(f'{MAP_ID}|{kind}|{label.casefold()}'.encode()).hexdigest()[:24]


def excerpt(text, position, width=460):
    start = max(0, position - 130)
    if start:
        space = text.find(' ', start, position)
        if space >= 0: start = space + 1
    end = min(len(text), start + width)
    if end < len(text):
        space = text.rfind(' ', start, end)
        if space > start: end = space
    return text[start:end].strip()



def source_fingerprint(documents,version='source-map-v3'):
    fingerprint=hashlib.sha256((version+'|'+CATALOG).encode())
    for record in documents:
        text=record.get('text') or ''
        if not isinstance(text,str) or not text.strip():continue
        metadata=record.get('metadata') or {}
        source=str(metadata.get('source') or metadata.get('filename') or 'Source collection')
        filename=str(metadata.get('filename') or source)
        fingerprint.update(json.dumps([text,source,filename,effective_year(metadata)],ensure_ascii=False).encode())
    return fingerprint.hexdigest()


def build_map(documents):
    nodes, passages, positions = {}, defaultdict(list), defaultdict(dict)
    root_id = node_id('Person', 'Jensen Huang')
    catalog = []
    for line in CATALOG.splitlines():
        category, kind, label, aliases = line.split('|')
        regex = re.compile(r'(?<!\w)(?:' + '|'.join(re.escape(a) for a in aliases.split(';')) + r')(?!\w)', re.I)
        catalog.append((category, kind, label, regex))
    sources = set()
    fingerprint = source_fingerprint(documents)
    for index, record in enumerate(documents):
        text = record.get('text') or ''
        if not isinstance(text, str) or not text.strip(): continue
        metadata = record.get('metadata') or {}
        source = str(metadata.get('source') or metadata.get('filename') or 'Source collection')
        filename = str(metadata.get('filename') or source)
        try: year = effective_year(metadata)
        except (TypeError, ValueError): year = 0
        sources.add(filename)
        for category, kind, label, regex in catalog:
            match = regex.search(text)
            if not match: continue
            key = node_id(kind, label)
            nodes.setdefault(key, {'id': key, 'label': label, 'kind': kind, 'category': category, 'mentions': 0})
            nodes[key]['mentions'] += 1
            passage = {'source': source, 'filename': filename, 'year': year,
                       'chunk_id': str(record.get('id', index)), 'quote': excerpt(text, match.start())}
            passage = provenance(metadata | passage)
            passages[key].append(passage)
            positions[index][key] = passage
    if not nodes:
        raise ValueError('No source-backed topics found in the corpus')
    if root_id not in nodes:
        nodes[root_id] = {'id': root_id, 'label': 'Jensen Huang', 'kind': 'Person', 'category': 'People & partners', 'mentions': len(documents)}
        # Root describes this source collection; it does not add biography claims.
        passages[root_id] = [p for values in positions.values() for p in list(values.values())[:1]][:3]
    categories = sorted({n['category'] for n in nodes.values()})
    entity_nodes = list(nodes.values())
    edges = []
    for category in categories:
        key = node_id('Category', category)
        nodes[key] = {'id': key, 'label': category, 'kind': 'Category', 'category': category, 'mentions': 0, 'evidence': []}
        edges.append({'source': root_id, 'target': key, 'label': 'COVERS', 'weight': 1, 'evidence': []})
    for node in entity_nodes:
        seen = set(); evidence = []
        for passage in sorted(passages[node['id']], key=lambda p: -p['year']):
            if passage['filename'] in seen: continue
            seen.add(passage['filename']); evidence.append(passage)
            if len(evidence) == 3: break
        node['evidence'] = evidence
        node['source_count'] = len({p['filename'] for p in passages[node['id']]})
        node['query'] = (f'Based on your source material, explain {node["label"]} and how it connects to your work at NVIDIA. Use a concise explanation and a concrete example.')
        if node['id'] != root_id:
            edges.append({'source': node_id('Category', node['category']), 'target': node['id'], 'label': 'IN_CATEGORY', 'weight': 1, 'evidence': []})
    pairs, pair_evidence = Counter(), {}
    for found in positions.values():
        for a, b in combinations(sorted(found), 2):
            if root_id in (a, b): continue
            pairs[a, b] += 1
            pair_evidence.setdefault((a, b), [found[a], found[b]])
    neighbors = defaultdict(list)
    for pair, count in pairs.items():
        for key in pair: neighbors[key].append((count, pair))
    selected_pairs = {pair for values in neighbors.values() for _, pair in sorted(values, reverse=True)[:3]}
    for a, b in sorted(selected_pairs):
        edges.append({'source': a, 'target': b, 'label': 'CO_MENTIONED', 'weight': pairs[a, b], 'evidence': pair_evidence[a, b]})
    return {'state': 'ready', 'map_id': MAP_ID, 'fingerprint': fingerprint, 'root_id': root_id,
            'document_count': len(documents), 'source_count': len(sources), 'categories': categories,
            'nodes': list(nodes.values()), 'edges': edges,
            'quality': corpus_quality(documents),
            'relationship_note': 'Topic connections mean the source passages mention both subjects; they are not claims of a business or personal relationship.'}


def corpus_quality(documents):
    hashes=Counter(); warnings=[]; undated=0; no_url=0
    for doc in documents:
        text=doc.get('text') or ''; metadata=doc.get('metadata') or {}
        hashes[hashlib.sha256(text.strip().encode()).hexdigest()]+=1
        source=provenance(metadata)
        if not effective_year(metadata):undated+=1
        if not source['url']:no_url+=1
        issues=[]
        if source.get('date_note'):issues.append(source['date_note'])
        if len(text.strip())<100:issues.append('Very short passage')
        if re.search(r'jensen wong|n\s?vidiaa|\[inaudible\]|\[unintelligible\]',text,re.I):issues.append('Possible transcription error; review original recording')
        if re.search(r'(.{40,200})\1\1',text):issues.append('Repeated transcript text')
        if issues:warnings.append({'chunk_id':doc.get('id'),'filename':source['filename'],'issues':issues})
    return {'duplicate_passages':sum(count-1 for count in hashes.values()),'undated_passages':undated,
            'passages_without_original_url':no_url,'review_flags':warnings[:50],
            'method':'Exact duplicate hashes, date/provenance checks, and transcript heuristics. Flagged text is preserved for review.'}


class KnowledgeMap:
    def __init__(self):
        self._lock = threading.RLock()
        self._snapshot = None
        self._state = 'building'
        self._neo4j_synced = False
        self._documents = []
        self._patterns = aliases(CATALOG)
        self._enrichment = 'waiting'
        self._enriching = False

    def initialize(self):
        with self._lock:
            if self._snapshot is None:
                try:
                    collection = get_chroma_collection()
                    documents = []
                    for offset in range(0, collection.count(), 500):
                        data = collection.get(limit=500, offset=offset, include=['documents', 'metadatas'])
                        documents.extend({'id': key, 'text': text, 'metadata': metadata}
                                         for key, text, metadata in zip(data['ids'], data['documents'], data['metadatas']))
                    repair_source_dates(collection,documents)
                    self._documents = documents
                    self._snapshot = build_map(documents)
                    self._state = 'ready'
                    logger.info('Jensen source map built: %d nodes from %d passages', len(self._snapshot['nodes']), len(documents))
                except Exception:
                    self._state = 'error'
                    logger.exception('Jensen source map could not be built')
                    return
            if graph.configured() and not self._neo4j_synced:
                try:
                    self._sync_neo4j()
                    self._neo4j_synced = True
                except Exception:
                    logger.exception('Source map Neo4j sync failed; keeping the corpus map available')

    def _sync_neo4j(self):
        # Introspect the actual labels before adding this separate public namespace.
        graph._query('CALL db.labels() YIELD label RETURN label', read=True)
        payload = self._snapshot
        nodes = [{k: v for k, v in node.items() if k not in ('evidence',)} |
                 {'evidence_json': json.dumps(node.get('evidence', []), ensure_ascii=False)} for node in payload['nodes']]
        edges = [edge | {'id': node_id('Edge', f'{edge["source"]}|{edge["label"]}|{edge["target"]}'),
                         'evidence_json': json.dumps(edge['evidence'], ensure_ascii=False)} for edge in payload['edges']]
        # One atomic, idempotent write, scoped to public source data only.
        graph._query('''
            UNWIND $nodes AS item
            MERGE (n:DTKnowledgeEntity {map_id: $map_id, id: item.id})
            SET n += item, n.fingerprint = $fingerprint, n.active = true
            WITH count(n) AS total
            UNWIND $edges AS item
            MATCH (s:DTKnowledgeEntity {map_id: $map_id, id: item.source})
            MATCH (t:DTKnowledgeEntity {map_id: $map_id, id: item.target})
            MERGE (s)-[r:DT_KNOWLEDGE_LINK {id: item.id}]->(t)
            SET r.predicate = item.label, r.weight = item.weight,
                r.evidence_json = item.evidence_json, r.fingerprint = $fingerprint, r.active = true
            WITH count(r) AS total
            MERGE (m:DTKnowledgeMap {map_id: $map_id})
            WITH m, m.fingerprint AS old_fingerprint, m.payload_json AS old_payload
            SET m.previous_payload_json = CASE WHEN old_fingerprint <> $fingerprint THEN old_payload ELSE m.previous_payload_json END,
                m.previous_relations_json = CASE WHEN old_fingerprint <> $fingerprint THEN m.relations_json ELSE m.previous_relations_json END,
                m.relations_json = CASE WHEN old_fingerprint <> $fingerprint THEN null ELSE m.relations_json END,
                m.fingerprint = $fingerprint, m.payload_json = $payload,
                m.updated_at = datetime()
            WITH m
            CALL { WITH m MATCH (n:DTKnowledgeEntity {map_id: $map_id}) WHERE (n.fingerprint IS NULL OR n.fingerprint <> $fingerprint) SET n.active=false RETURN count(n) AS retired_nodes }
            CALL { WITH m MATCH (a:DTKnowledgeEntity {map_id: $map_id})-[r:DT_KNOWLEDGE_LINK]->() WHERE (r.fingerprint IS NULL OR r.fingerprint <> $fingerprint) SET r.active=false RETURN count(r) AS retired_edges }
            RETURN retired_nodes, retired_edges
        ''', {'nodes': nodes, 'edges': edges, 'map_id': MAP_ID,
              'fingerprint': payload['fingerprint'], 'payload': json.dumps(payload, ensure_ascii=False)})

    def snapshot(self):
        if self._snapshot is None:
            return {'state': self._state, 'nodes': [], 'edges': [],
                    'error': 'The source map could not load. Refresh to try again.' if self._state == 'error' else None}
        return self._snapshot | {'neo4j_synced': self._neo4j_synced, 'relation_state': self._enrichment, 'semantic_relation_count': sum(e['label'] in PREDICATES for e in self._snapshot['edges'])}

    def context(self, query, era='all', node_id=None):
        snapshot = self.snapshot()
        if snapshot.get('state') != 'ready': return {'paths': [], 'evidence': [], 'backend': 'none'}
        seeds = [n['id'] for n in snapshot['nodes'] if n['kind'] != 'Category' and self._patterns[n['label']].search(query)]
        if node_id and node_id not in seeds: seeds.insert(0, node_id)
        paths, backend = retrieve_paths(snapshot, seeds[:4], era)
        evidence = [e for path in paths for edge in path['edges'] for e in edge['evidence']]
        for key in seeds[:4]:
            node = self.node(key)
            evidence.extend(e for e in node.get('evidence', []) if in_era(e.get('year'), era))
        return {'paths': paths, 'evidence': evidence[:20], 'backend': backend}

    def timeline(self):
        years = defaultdict(lambda: {'sources': set(), 'passages': 0, 'topics': Counter()})
        for doc in self._documents:
            metadata = doc.get('metadata') or {}; year = effective_year(metadata)
            if not year: continue
            years[year]['sources'].add(metadata.get('filename') or metadata.get('source') or 'Corpus')
            years[year]['passages'] += 1
            for label, pattern in self._patterns.items():
                if pattern.search(doc['text']): years[year]['topics'][label] += 1
        return [{'year': year, 'sources': sorted(item['sources']), 'passages': item['passages'],
                 'topics': [label for label, _ in item['topics'].most_common(8)]} for year, item in sorted(years.items())]

    def enrich(self):
        # A bounded source-only job, separate from chat and visitor-memory updates.
        with self._lock:
            if self._enriching or not self._snapshot or self._enrichment == 'ready': return
            self._enriching = True; self._enrichment = 'building'
        try:
            if graph.configured():
                rows = graph._query('MATCH (m:DTKnowledgeMap {map_id:$map_id, fingerprint:$fingerprint}) RETURN m.relations_json AS relations,m.previous_payload_json AS previous,m.previous_relations_json AS previous_relations',
                    {'map_id': MAP_ID, 'fingerprint': self._snapshot['fingerprint']}, read=True)
                cached=json.loads(rows[0]['relations']) if rows and rows[0].get('relations') else None
                migrated=False
                if cached is None and rows and rows[0].get('previous'):
                    previous=json.loads(rows[0]['previous'])
                    # Upgrade the validator over the same corpus without another paid model pass.
                    if previous.get('fingerprint')==source_fingerprint(self._documents,'source-map-v2'):
                        cached=json.loads(rows[0]['previous_relations']) if rows[0].get('previous_relations') else [edge for edge in previous['edges'] if edge['label'] in PREDICATES];migrated=True
                # Shipped public-corpus evidence is accepted only for this exact content version.
                if not cached:
                    from pathlib import Path
                    path=Path('evals/source-relations.json')
                    if path.exists():
                        bundled=json.loads(path.read_text())
                        if bundled.get('corpus_fingerprint')==self._snapshot['fingerprint']:
                            cached=bundled['relations'];migrated=True
                if cached is not None:
                    payload={'relations':[{'source_id':edge['source'],'target_id':edge['target'],'relation':edge['label'],
                        'chunk_id':quote['chunk_id'],'evidence':quote['quote']} for edge in cached for quote in edge['evidence']]}
                    relations=validate_relations(payload,self._snapshot['nodes'],self._documents,self._patterns)
                    self._snapshot = self._snapshot | {'edges': [e for e in self._snapshot['edges'] if e['label'] not in PREDICATES] + relations}
                    # Startup's base-map sync retires old links; reactivate validated cached links too.
                    self._sync_neo4j();self._neo4j_synced=True
                    graph._query('MATCH (m:DTKnowledgeMap {map_id:$map_id, fingerprint:$fingerprint}) SET m.relations_json=$relations',
                        {'map_id':MAP_ID,'fingerprint':self._snapshot['fingerprint'],'relations':json.dumps(relations,ensure_ascii=False)})
                    self._enrichment = 'ready'; return
            if not self._documents: self._enrichment = 'unavailable'; return
            # Round-robin source sampling gives every document/year a chance.
            grouped = defaultdict(list)
            for doc in self._documents: grouped[(doc.get('metadata') or {}).get('filename', '')].append(doc)
            ordered = []
            while any(grouped.values()):
                for values in grouped.values():
                    if values: ordered.append(values.pop(0))
            batches = max(0, min(60, int(os.getenv('MAP_RELATION_BATCHES', '8'))))
            relations, seen, failures = [], set(), 0
            for start in range(0, min(len(ordered), batches * 6), 6):
                try: extracted = extract_batch(ordered[start:start+6], self._snapshot['nodes'], CATALOG)
                except Exception:
                    failures += 1; logger.warning('Source relation batch unavailable')
                    if failures >= 2: break
                    continue
                for edge in extracted:
                    key = (edge['source'], edge['target'], edge['label'])
                    if key not in seen: seen.add(key); relations.append(edge)
            if failures and not relations: self._enrichment = 'retry_needed'; return
            self._snapshot = self._snapshot | {'edges': [e for e in self._snapshot['edges'] if e['label'] not in PREDICATES] + relations}
            self._enrichment = 'ready' if not failures else 'partial'
            if graph.configured():
                self._sync_neo4j(); self._neo4j_synced = True
                graph._query('MATCH (m:DTKnowledgeMap {map_id:$map_id, fingerprint:$fingerprint}) SET m.relations_json=$relations',
                    {'map_id':MAP_ID,'fingerprint':self._snapshot['fingerprint'],'relations':json.dumps(relations,ensure_ascii=False)})
        except Exception:
            logger.exception('Source enrichment failed; base map remains usable'); self._enrichment = 'retry_needed'
        finally: self._enriching = False

    def node(self, key):
        if self._snapshot:
            return next((node for node in self._snapshot['nodes'] if node['id'] == key), None)
        return None


knowledge_map = KnowledgeMap()
