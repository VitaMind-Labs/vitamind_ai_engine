"""Audit/normalize the supplied FAQ CSV; never invent planning labels."""
import collections,csv,hashlib,io,json,random,re
from pathlib import Path
from lumina.adhd.executive_function.language import normalize

ROOT=Path(__file__).resolve().parents[1]

def main():
    source=ROOT/'data/raw/adhd_dataset.csv'
    raw=source.read_bytes()
    try: decoded=raw.decode('utf-8-sig'); encoding='utf-8-sig'
    except UnicodeDecodeError: decoded=raw.decode('cp1252'); encoding='cp1252'
    rows=list(csv.DictReader(io.StringIO(decoded,newline='')))
    valid=[]; quarantined=[]; answers=collections.defaultdict(set)
    for i,row in enumerate(rows,2):
        question=(row.get('Question') or '').strip(); answer=(row.get('Answer') or '').strip()
        errors=[]
        if not question or not answer: errors.append('EMPTY_QUESTION_OR_ANSWER')
        if (row.get('') or '').strip() or None in row: errors.append('UNNAMED_COLUMN_HAS_CONTENT')
        if errors:
            quarantined.append({'sourceRow':i,'reasons':errors,'original':row}); continue
        record={'id':f'faq-{i}','sourceRow':i,'question':question,'answer':answer,'normalizedQuestion':normalize(question),'language':'EN','usage':'UNREVIEWED_REFERENCE_ONLY'}
        answers[record['normalizedQuestion']].add(normalize(answer))
        valid.append(record)
    # All answers for an exact question stay together; multiple answers are flagged,
    # not automatically declared contradictions or used as diagnosis/training labels.
    questions=sorted({r['normalizedQuestion'] for r in valid})
    parents=list(range(len(questions)))
    def find(i):
        while parents[i]!=i: parents[i]=parents[parents[i]]; i=parents[i]
        return i
    def union(a,b): parents[find(a)]=find(b)
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors
    vec=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),min_df=1,max_features=40000)
    x=vec.fit_transform(questions)
    nn=NearestNeighbors(n_neighbors=min(8,len(questions)),metric='cosine',algorithm='brute',n_jobs=1).fit(x)
    links=0
    for start in range(0,len(questions),256):
        distances,neighbors=nn.kneighbors(x[start:start+256])
        for offset,(ds,ns) in enumerate(zip(distances,neighbors)):
            for distance,j in zip(ds,ns):
                i=start+offset
                if i!=j and distance<=.06:
                    union(i,int(j)); links+=1
    group={q:f'faq-family-{find(i)}' for i,q in enumerate(questions)}
    families=sorted(set(group.values())); random.Random(42).shuffle(families)
    n=len(families); train=set(families[:int(n*.7)]); val=set(families[int(n*.7):int(n*.85)])
    splits={name:[] for name in ('train','validation','test')}
    for r in valid:
        r['groupId']=group[r['normalizedQuestion']]
        r['multipleAnswerVariants']=len(answers[r['normalizedQuestion']])>1
        split='train' if r['groupId'] in train else 'validation' if r['groupId'] in val else 'test'
        splits[split].append(r)
    for folder in ('normalized','splits'): (ROOT/'data'/folder).mkdir(exist_ok=True)
    def write(path,records): path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records),encoding='utf-8')
    write(ROOT/'data/normalized/faq.jsonl',valid)
    write(ROOT/'data/normalized/quarantine.jsonl',quarantined)
    for name,rs in splits.items(): write(ROOT/f'data/splits/faq_{name}.jsonl',rs)
    report={'sourceSha256':hashlib.sha256(raw).hexdigest(),'encoding':encoding,'rawRows':len(rows),'validReferenceRows':len(valid),'quarantinedRows':len(quarantined),'rowsBySplit':{k:len(v) for k,v in splits.items()},'nearDuplicateGrouping':{'method':'exact normalized questions plus character-TFIDF nearest-neighbor graph','cosineSimilarityMinimum':.94,'neighborsPerQuestion':8,'links':links,'groups':len(families),'limitation':'Approximate lexical grouping; not proof that all semantic paraphrases are separated.'},'intentLabels':0,'frictionLabels':0,'taskAnnotations':0,'arabicRows':0,'planningModelTraining':'BLOCKED_MISSING_LABELS','splitUsage':'QA reference/audit only. Not a held-out planning benchmark. No FAQ answers are returned at runtime.'}
    (ROOT/'reports/dataset-preparation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
