"""Small, auditable metrics. No model-as-judge, keyword answer grading, or secrets."""
import hashlib
import json
import math
import statistics


def mean(values):
    values = [v for v in values if v is not None]
    return round(statistics.mean(values), 4) if values else None


def percentile(values, fraction):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    position = (len(values) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    return round(values[low] + (values[high] - values[low]) * (position-low), 4)


def rank_metrics(case, ids, k=5):
    grades = {p['chunk_id']: p['grade'] for p in case['relevant_passages']}
    if not grades:
        return {'recall': None, 'mrr': None, 'ndcg': None}
    ranked = list(dict.fromkeys(ids))[:k]
    hits = [grades.get(key, 0) for key in ranked]
    dcg = sum((2**grade - 1) / math.log2(rank+2) for rank, grade in enumerate(hits))
    ideal = sum((2**grade - 1) / math.log2(rank+2) for rank, grade in enumerate(sorted(grades.values(), reverse=True)[:k]))
    return {'recall': sum(key in grades for key in ranked) / len(grades),
            'mrr': next((1/(rank+1) for rank, grade in enumerate(hits) if grade > 0), 0),
            'ndcg': dcg/ideal if ideal else 0}


def sample_digest(answer, citations):
    content=json.dumps({'answer':answer,'citations':citations},sort_keys=True,ensure_ascii=False)
    return hashlib.sha256(content.encode()).hexdigest()


def review_metrics(samples, reviews):
    """Only score explicit reviewed decisions attached to the exact response."""
    indexed={(r['question_id'],r['strategy']):r for r in reviews}
    audited=[]
    for sample in samples:
        review=indexed.get((sample['question_id'],sample['strategy']))
        if not review or not sample['completed'] or review.get('response_digest')!=sample['response_digest']:
            continue
        if not isinstance(review.get('abstained'),bool):
            continue
        audited.append((sample,review))
    abstentions=[(s,r) for s,r in audited if s['expected_behavior']=='abstain']
    answerable=[(s,r) for s,r in audited if s['expected_behavior']!='abstain']
    misleading=[(s,r) for s,r in audited if s['expected_behavior']=='correct_premise']
    claims=[claim for _,r in audited for claim in r.get('claims',[]) if claim.get('support') in {'supported','unsupported','uncited'}]
    cited=[c for c in claims if c.get('citation_ids')]
    return {'reviewed_responses':len(audited),
            'abstention_recall':mean([int(r['abstained']) for _,r in abstentions]),
            'false_abstention_rate':mean([int(r['abstained']) for _,r in answerable]),
            'premise_correction_rate':mean([int(r['corrected_premise']) for _,r in misleading if isinstance(r.get('corrected_premise'),bool)]),
            'citation_support_precision':mean([int(c['support']=='supported') for c in cited]),
            'claim_citation_coverage':len(cited)/len(claims) if claims else None,
            'reviewed_factual_claims':len(claims), 'reviewed_cited_claims':len(cited)}
