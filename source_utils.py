"""Shared era boundaries and source provenance; no model calls."""
import json
from pathlib import Path
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse

ERA_YEAR_RANGES = {'all': (None, None), 'pre_cuda': (None, 2007), 'deep_learning': (2012, 2021), 'llm_era': (2022, None)}
CATALOG = json.loads(Path(__file__).with_name('source_catalog.json').read_text())


# This shipped biography is tagged 2007 but includes 2025/2026 events.
# Preserve it for general answers; do not fabricate a publication date.
UNVERIFIED_SOURCE_DATES={'jensen_personal.txt': 'The shipped biography has mixed dates through 2026, despite its original 2007 label.'}

def effective_year(record):
    if record.get('filename') in UNVERIFIED_SOURCE_DATES:return 0
    if record.get('filename') == 'milken_2026.txt':return 2025
    try:return int(record.get('year') or 0)
    except (TypeError,ValueError):return 0


def in_era(year, era='all'):
    low, high = ERA_YEAR_RANGES.get(era, (None, None))
    try: year = int(year or 0)
    except (TypeError, ValueError): year = 0
    return era == 'all' or (year > 0 and (low is None or year >= low) and (high is None or year <= high))


def provenance(record):
    filename = str(record.get('filename') or record.get('source') or 'Included corpus')
    registered = CATALOG.get(filename, {})
    url = record.get('url') or registered.get('url')
    if not isinstance(url, str) or urlparse(url).scheme not in ('https', 'http'): url = None
    start = record.get('start_seconds', record.get('timestamp_start'))
    try: start = max(0, int(float(start))) if start is not None else None
    except (TypeError, ValueError): start = None
    if url and start is not None and 'youtube.com' in (urlparse(url).hostname or ''):
        parsed = urlparse(url); query = dict(parse_qsl(parsed.query)); query['t'] = str(start)
        url = urlunparse(parsed._replace(query=urlencode(query)))
    return {'filename': filename, 'source': str(record.get('source') or registered.get('title') or filename),
            'title': registered.get('title') or filename.removesuffix('.txt').replace('_', ' ').upper(),
            'year': effective_year(record), 'date_note': UNVERIFIED_SOURCE_DATES.get(filename) or registered.get('date_note'), 'url': url, 'start_seconds': start,
            'chunk_id': str(record.get('chunk_id', record.get('chunk_index', ''))),
            'quote': str(record.get('quote') or record.get('text') or '')}


def citations_for(records, era='all', limit=12):
    citations, seen = [], set()
    for record in records:
        if not in_era(effective_year(record), era): continue
        item = provenance(record)
        key = (item['filename'], item['chunk_id'], item['quote'][:100])
        if not item['quote'] or key in seen: continue
        seen.add(key); item['id'] = len(citations) + 1; citations.append(item)
        if len(citations) >= limit: break
    return citations


def repair_source_dates(collection, documents):
    ids=[];metadatas=[]
    for doc in documents:
        metadata=doc.get('metadata') or {};year=effective_year(metadata)
        if metadata.get('filename') in {'jensen_personal.txt','milken_2026.txt'} and metadata.get('year')!=year:
            changed=metadata | {'year':year,'declared_year':int(metadata.get('year') or 0)}
            doc['metadata']=changed;ids.append(doc['id']);metadatas.append(changed)
    if ids:collection.update(ids=ids,metadatas=metadatas)


def focus_excerpt(text, query, limit=500):
    """Choose a contiguous source window; preserve the quotation verbatim."""
    import re
    if len(text)<=limit:return text
    ignored={'explain','about','what','does','have','said','jensen','source','material','using','your','with','from','which','that','this','review','compare','year','years','relationship','between'}
    words={word for word in re.findall(r'\b[\w-]{3,}\b',query.lower()) if word not in ignored}
    matches=list(re.finditer(r'\b(?:'+ '|'.join(re.escape(word) for word in words)+r')\b',text,re.I))[:60] if words else []
    starts={0} | {max(0,match.start()-100) for match in matches}
    start=max(starts,key=lambda point:(sum(len(re.findall(r'\b'+re.escape(word)+r'\b',text[point:point+limit],re.I)) for word in words),-point))
    if start>0:
        space=text.find(' ',start,min(start+30,len(text)))
        if space>=0:start=space+1
    end=min(len(text),start+limit)
    if end<len(text):
        space=text.rfind(' ',max(start,end-40),end)
        if space>start:end=space
    return text[start:end]


def normalize_citations(answer, allowed):
    import re
    def clean(match):
        body=match[1]
        if re.fullmatch(r'\s*\d+(?:\s*,\s*\d+)*\s*',body):
            return ''.join(f'[{int(number)}]' for number in re.findall(r'\d+',body) if int(number) in allowed)
        if 'map_' in body or 'CO_MENTIONED' in body:
            leading=re.match(r'\s*(\d+(?:\s*,\s*\d+)*)',body)
            return ''.join(f'[{int(number)}]' for number in re.findall(r'\d+',leading[1]) if int(number) in allowed) if leading else ''
        return match[0]
    return re.sub(r'\[([^\]\n]+)\]',clean,answer).strip()
