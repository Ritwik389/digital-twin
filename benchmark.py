"""Reproducible public-corpus checks; never reads visitor history or credentials.
python benchmark.py --base https://rj8307-digital-twin-2.hf.space
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import statistics
import time
import uuid
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

HTTP=requests.Session()
HTTP.mount('https://',HTTPAdapter(max_retries=Retry(total=3,connect=3,read=0,status=0,backoff_factor=.5)))


def run(base, output, reuse_retrieval=None, generation_interval=15):
    cases=json.loads(Path('evals/questions.json').read_text())
    corpus=HTTP.get(base+'/api/corpus/status',timeout=60).json()
    previous=json.loads(reuse_retrieval.read_text()) if reuse_retrieval else None
    retrieval=previous['retrieval_results'] if previous else []
    for case in ([] if previous else cases):
        start=time.perf_counter()
        response=HTTP.post(base+'/api/retrieval/inspect',json={'query':case['query'],'era':case.get('era','all')},timeout=180)
        response.raise_for_status();data=response.json()
        citations=data['citations'];text=' '.join(c['quote'] for c in citations).lower()
        hit=any(alias.lower() in text for alias in case['expected_aliases'])
        era=case.get('era','all')
        bounds={'all':(0,2100),'pre_cuda':(1,2007),'deep_learning':(2012,2021),'llm_era':(2022,2100)}[era]
        valid=all(bounds[0]<=c['year']<=bounds[1] for c in citations)
        retrieval.append(case | {'topic_hit':hit,'era_valid':valid,'source_count':len(citations),'latency_seconds':round(time.perf_counter()-start,3)})
        print(f'Retrieval {len(retrieval)}/{len(cases)}: topic_hit={hit} era_valid={valid}',flush=True)
    generation=[];run_id=uuid.uuid4().hex[:12]
    samples=[
        {'query':'Hi Jensen, explain CUDA and why software matters.','mode':'beginner'},
        {'query':'Explain the relationship between Omniverse and digital twins.'},
        {'query':'How do Blackwell and NVLink connect in AI infrastructure?'},
        {'query':'Review my startup pitch: a simulation platform for robotics teams.','mode':'pitch'},
        {'query':'Review an architecture with a GPU inference service, request batching, and a CPU API server.','mode':'architecture'},
        {'query':'Compare the source coverage of accelerated computing in 2019 and 2025.','mode':'compare','compare_years':[2019,2025]},
        {'query':'Explain Blackwell as it was understood in 2007.','era':'pre_cuda'},
        {'query':'What is the secret unpublished NVIDIA chip launch date next month?'},
    ]
    for index,case in enumerate(samples):
        if index and generation_interval: time.sleep(generation_interval)
        visitor=f'evaluation_{run_id}_{index}'
        start=time.perf_counter();first=None;deltas=0;done=None;error=None
        try:
            with HTTP.post(base+'/api/chat/stream',json={'user_id':visitor,'session_id':'evaluation','message':case['query'],'era':case.get('era','all'),'mode':case.get('mode','conversation'),'compare_years':case.get('compare_years'),'voice_enabled':False},stream=True,timeout=(10,180)) as response:
                response.raise_for_status()
                for line in response.iter_lines(chunk_size=1):
                    if not line:continue
                    event=json.loads(line)
                    if event['type']=='delta':
                        deltas+=1
                        if first is None:first=time.perf_counter()-start
                    elif event['type']=='done':done=event
                    elif event['type']=='error':error=event['error']
            elapsed=time.perf_counter()-start
            answer=done['answer'] if done else ''
            ids={c['id'] for c in (done or {}).get('citations',[])}
            references=[int(s) for s in re.findall(r'\[(\d+)\]',answer)]
            generation.append(case | {'completed':bool(done),'answer':answer,'citations':(done or {}).get('citations',[]),
                'graph_paths_used':len((done or {}).get('graph_paths',[])),'graph_backend':(done or {}).get('graph_backend'),
                'citation_ids_valid':bool(references) and all(key in ids for key in references),'has_citation':bool(references),
                'first_token_seconds':round(first,3) if first is not None else None,'total_seconds':round(elapsed,3),'delta_count':deltas,'error':error})
            print(f'Generation {index+1}/{len(samples)}: complete={bool(done)} first_token={first}',flush=True)
        finally:
            # Only this run's disposable evaluation visitor is removed.
            response=HTTP.delete(base+'/api/memory/'+visitor,timeout=60)
            if not response.ok:print('Evaluation visitor cleanup did not complete',flush=True)
    times=[g['first_token_seconds'] for g in generation if g['first_token_seconds'] is not None]
    cited=[g for g in generation if g['has_citation']]
    report={'state':'complete','run_date':datetime.now(timezone.utc).isoformat(timespec='seconds'),'corpus_fingerprint':corpus.get('fingerprint'),
        'retrieval_cases':len(retrieval),'generation_cases':len(generation),
        'retrieval_topic_hit_rate':round(100*sum(r['topic_hit'] for r in retrieval)/len(retrieval),1),
        'era_filter_pass_rate':round(100*sum(r['era_valid'] for r in retrieval)/len(retrieval),1),
        'citation_validity_rate':round(100*sum(g['citation_ids_valid'] for g in cited)/len(cited),1) if cited else 0,
        'answers_with_citations':len(cited),'completed_generation_cases':sum(g['completed'] for g in generation),
        'first_token_p50_seconds':round(statistics.median(times),2) if times else None,
        'total_response_p50_seconds':round(statistics.median([g['total_seconds'] for g in generation]),2),
        'method':'Fifty fixed retrieval questions: an expected topic alias must appear in at least one of the top five retrieved passages. Every returned date is checked against the selected era. Eight streamed generation samples check completion, resolvable numbered citations, graph use, and client-observed latency; voice is disabled.',
        'limitations':'Topic matches do not measure answer correctness. Citation validity checks IDs, not whether every claim is entailed. Latency reflects this one live run after retrieval warm-up and varies with provider load. This is not an overall accuracy score.',
        'retrieval_run_date':previous.get('run_date') if previous else datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'retrieval_reused_from_unchanged_pipeline':bool(previous),
        'generation_cooldown_seconds':generation_interval,
        'retrieval_results':retrieval,'generation_results':generation}
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print(json.dumps({key:report[key] for key in ('retrieval_topic_hit_rate','era_filter_pass_rate','citation_validity_rate','first_token_p50_seconds','completed_generation_cases')}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--base',required=True);parser.add_argument('--output',type=Path,default=Path('reports/benchmark.json'))
    parser.add_argument('--reuse-retrieval',type=Path,help='Reuse fixed retrieval checks when the corpus and retrieval pipeline are unchanged; generation is always rerun.')
    parser.add_argument('--generation-interval',type=float,default=15,help='Cooldown between generation cases, excluded from response timing.')
    args=parser.parse_args();run(args.base.rstrip('/'),args.output,args.reuse_retrieval,max(0,args.generation_interval))
