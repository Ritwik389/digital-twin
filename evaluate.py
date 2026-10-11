"""Paired source-grounding evaluation. Run against a server with the same corpus.

python evaluate.py --base http://localhost:7860 --generation
python evaluate.py --score-manual --reviews evals/manual_reviews.json
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time
import uuid

import requests

from evaluation_metrics import mean, percentile, rank_metrics, review_metrics, sample_digest

STRATEGIES=('vector','reranked')


def write_json(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


def summaries(report, reviews):
    result={}
    for strategy in STRATEGIES:
        retrieval=[r for r in report['retrieval_results'] if r['strategy']==strategy]
        generation=[g for g in report['generation_results'] if g['strategy']==strategy]
        completed=[g for g in generation if g['completed']]
        refs=[valid for g in completed for valid in g['citation_ids_valid']]
        result[strategy]={
            'retrieval_questions':len(retrieval),
            'labeled_positive_questions':sum(r['recall_at_5'] is not None for r in retrieval),
            'recall_at_5':mean([r['recall_at_5'] for r in retrieval]),
            'candidate_recall_at_15':mean([r['candidate_recall_at_15'] for r in retrieval]),
            'mrr_at_5':mean([r['mrr_at_5'] for r in retrieval]),
            'ndcg_at_5':mean([r['ndcg_at_5'] for r in retrieval]),
            'era_filter_pass_rate':mean([int(r['era_valid']) for r in retrieval]),
            'retrieval_client_p50_ms':percentile([r['client_ms'] for r in retrieval],.5),
            'retrieval_client_p95_ms':percentile([r['client_ms'] for r in retrieval],.95),
            'ranking_server_p50_ms':percentile([r['ranking_ms'] for r in retrieval],.5),
            'generation_requests':len(generation),'completed_responses':len(completed),
            'generation_errors':len(generation)-len(completed),
            'first_text_client_p50_seconds':percentile([g['first_text_seconds'] for g in completed],.5),
            'first_text_client_p95_seconds':percentile([g['first_text_seconds'] for g in completed],.95),
            'complete_reply_client_p50_seconds':percentile([g['total_seconds'] for g in completed],.5),
            'citation_id_validity':mean([int(v) for v in refs]),
            **review_metrics(generation,reviews),
        }
    return result


def run(args):
    dataset_bytes=args.dataset.read_bytes()
    dataset=json.loads(dataset_bytes)
    cases=dataset['questions']
    session=requests.Session()
    base=args.base.rstrip('/')
    # Verify the app is reachable. Each generation has a fresh disposable ID;
    # no real visitor history is read.
    session.get(base+'/',timeout=30).raise_for_status()
    corpus=session.get(base+'/api/corpus/status',timeout=60).json()
    if corpus.get('state')!='ready':
        raise RuntimeError('Wait for the source map to become ready before evaluation.')
    if dataset.get('corpus_fingerprint')!=corpus.get('fingerprint'):
        raise RuntimeError('Corpus fingerprint differs from the labeled dataset. Relabel before scoring.')
    report={'schema_version':1,'state':'running','run_date':datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'base_url':base,'corpus_fingerprint':corpus['fingerprint'],
            'dataset_sha256':hashlib.sha256(dataset_bytes).hexdigest(),'dataset_label_status':dataset['label_status'],
            'pipeline_version':'grounded-eval-v1','server_revision':args.server_revision,'candidate_pool':15,'top_k':5,
            'category_counts':dict(Counter(c['category'] for c in cases)),
            'graph_enabled':False,'voice_enabled':False,'generation_cooldown_seconds':args.generation_interval,
            'warmup':[],'retrieval_results':[],'generation_results':[],
            'method':'Paired Chroma top-5 versus a cross-encoder rerank of the same top-15 pool. Passage labels identify checked supporting excerpts. Generation uses 16 balanced cases, each in a new disposable conversation, with graph and voice disabled. Case/strategy order alternates. Cold initialization is recorded separately.',
            'limitations':'Passage labels are provisional coding-agent annotations, not independent human labels, and are not exhaustive. Recall measures annotated passage coverage. One corpus, one client/server setup and one generated response per case/strategy; not overall answer accuracy. Citation ID validity is distinct from claim support. Errors are not counted as successful abstentions. Claim-support and behavior scores require explicit response-matched review decisions.'}
    for strategy in STRATEGIES:
        start=time.perf_counter()
        response=session.post(base+'/api/retrieval/inspect',json={'query':cases[0]['question'],'strategy':strategy,'strict_reranker':True},timeout=240)
        response.raise_for_status()
        data=response.json()
        report['warmup'].append({'strategy':strategy,'client_seconds':round(time.perf_counter()-start,3),'ranking_backend':data['ranking_backend']})
    for index,case in enumerate(cases):
        order=STRATEGIES if index%2==0 else STRATEGIES[::-1]
        for strategy in order:
            start=time.perf_counter()
            response=session.post(base+'/api/retrieval/inspect',json={'query':case['question'],'era':case['era'],'strategy':strategy,'strict_reranker':True},timeout=180)
            response.raise_for_status();data=response.json()
            if strategy=='reranked' and data['citations'] and data['ranking_backend']!='cross_encoder':
                raise RuntimeError('Reranker fallback cannot be scored as a reranked pipeline.')
            ids=[c['chunk_id'] for c in data['citations']]
            metrics=rank_metrics(case,ids)
            bounds={'all':(None,None),'pre_cuda':(1,2007),'deep_learning':(2012,2021),'llm_era':(2022,None)}[case['era']]
            valid=all(case['era']=='all' or (c['year']>0 and (bounds[0] is None or c['year']>=bounds[0]) and (bounds[1] is None or c['year']<=bounds[1])) for c in data['citations'])
            report['retrieval_results'].append({'question_id':case['id'],'category':case['category'],'strategy':strategy,
                'ranking_backend':data['ranking_backend'],'ranked_chunk_ids':ids,'candidate_chunk_ids':data['candidate_ids'],
                'recall_at_5':metrics['recall'],'mrr_at_5':metrics['mrr'],'ndcg_at_5':metrics['ndcg'],
                'candidate_recall_at_15':rank_metrics(case,data['candidate_ids'],15)['recall'],
                'era_valid':valid,'empty_results':not ids,'server_ms':data['retrieval_ms'],'ranking_ms':data['ranking_ms'],
                'client_ms':round((time.perf_counter()-start)*1000,3)})
        pair=report['retrieval_results'][-2:]
        if pair[0]['candidate_chunk_ids'] != pair[1]['candidate_chunk_ids']:
            raise RuntimeError('The paired candidate pools differ; this run cannot isolate reranking.')
        write_json(args.output,report)
        print(f"Retrieval {index+1}/{len(cases)}",flush=True)
    run_id=uuid.uuid4().hex
    subset=[c for c in cases if c['manual_review_subset']] if args.generation else []
    for index,case in enumerate(subset):
        order=STRATEGIES if index%2==0 else STRATEGIES[::-1]
        for strategy in order:
            if report['generation_results'] and args.generation_interval:
                time.sleep(args.generation_interval)
            visitor=f'evaluation_{run_id}_{case["id"]}_{strategy}'
            start=time.perf_counter();first=None;done=None;error=None
            try:
                with session.post(base+'/api/chat/stream',json={'user_id':visitor,'session_id':'evaluation','message':case['question'],'era':case['era'],'voice_enabled':False,'use_graph':False,'retrieval_strategy':strategy},stream=True,timeout=(10,180)) as response:
                    response.raise_for_status()
                    for line in response.iter_lines(chunk_size=1):
                        if not line:continue
                        event=json.loads(line)
                        if event['type']=='delta' and first is None:first=time.perf_counter()-start
                        if event['type']=='done':done=event
                        if event['type']=='error':error=event.get('error')
            except (requests.RequestException,ValueError):
                error='Transport or response failure'
            elapsed=time.perf_counter()-start
            answer=(done or {}).get('answer','');citations=(done or {}).get('citations',[])
            allowed={c['id'] for c in citations};references=[int(i) for i in re.findall(r'\[(\d+)\]',answer)]
            completed=bool(done) and done.get('completion_status')!='retrieval_unavailable'
            sample={'question_id':case['id'],'category':case['category'],'expected_behavior':case['expected_behavior'],
                    'strategy':strategy,'completed':completed,'answer':answer,'citations':citations,
                    'response_digest':sample_digest(answer,citations),'error':error,
                    'ranking_backend':(done or {}).get('ranking_backend'),'generation_path':(done or {}).get('generation_path'),
                    'first_text_seconds':round(first,4) if first is not None else None,'total_seconds':round(elapsed,4),
                    'citation_ids_valid':[i in allowed for i in references]}
            if strategy=='reranked' and citations and sample['ranking_backend']!='cross_encoder':
                sample['completed']=False;sample['error']='Cross-encoder fell back during generation'
            report['generation_results'].append(sample)
            try:
                cleanup=session.delete(base+'/api/memory/'+visitor,timeout=60)
                sample['cleanup_ok']=cleanup.ok
            except requests.RequestException:sample['cleanup_ok']=False
            write_json(args.output,report)
            print(f"Generation {len(report['generation_results'])}/{2*len(subset)}: {case['id']} {strategy}, complete={sample['completed']}",flush=True)
    report['state']='complete' if args.generation else 'retrieval_only'
    report['manual_review_status']='pending'
    report['summary']=summaries(report,[])
    write_json(args.output,report)
    template={'status':'pending_human_review','reviews':[{'question_id':s['question_id'],'strategy':s['strategy'],'response_digest':s['response_digest'],'reviewer':None,'reviewer_type':None,'abstained':None,'corrected_premise':None,'claims':[],'notes':''} for s in report['generation_results']]}
    write_json(args.output.with_name('manual-review-template.json'),template)
    print(json.dumps(report['summary'],indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--base')
    parser.add_argument('--server-revision',help='Verified deployment commit, recorded as provenance')
    parser.add_argument('--dataset',type=Path,default=Path('evals/grounded_questions.json'))
    parser.add_argument('--output',type=Path,default=Path('reports/evaluation.json'))
    parser.add_argument('--generation',action='store_true')
    parser.add_argument('--generation-interval',type=float,default=15)
    parser.add_argument('--score-manual',action='store_true')
    parser.add_argument('--reviews',type=Path,default=Path('evals/manual_reviews.json'))
    args=parser.parse_args()
    if args.score_manual:
        report=json.loads(args.output.read_text());review_file=json.loads(args.reviews.read_text())
        report['summary']=summaries(report,review_file['reviews'])
        report['manual_review_status']=review_file['status']
        write_json(args.output,report)
    else:
        if not args.base:parser.error('--base is required unless scoring existing manual reviews')
        if args.generation_interval<0:parser.error('generation interval must be nonnegative')
        run(args)
