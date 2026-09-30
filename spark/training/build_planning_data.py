"""Build traceable public + synthetic planning data, without labeling FAQ answers."""
import collections,csv,hashlib,json,random,re
from pathlib import Path
from lumina.adhd.executive_function.language import normalize,languages

ROOT=Path(__file__).resolve().parents[1]
PRIORITY={'train':0,'validation':1,'test':2}
OBJECTS=[
    ('email James','emailing James','أرسل رسالة إلى خالد','إرسال رسالة إلى خالد'),
    ('call the bank','calling the bank','أتصل بالبنك','الاتصال بالبنك'),
    ('write the report','writing the report','أكتب التقرير','كتابة التقرير'),
    ('buy groceries','buying groceries','أشتري أغراض البيت','شراء أغراض البيت'),
    ('review the notes','reviewing the notes','أراجع الملاحظات','مراجعة الملاحظات'),
    ('clean the desk','cleaning the desk','أنظف المكتب','تنظيف المكتب'),
]

def observed_language(text):
    found=languages(text)
    return 'MIXED' if len(found)>1 else found[0] if found else 'UNKNOWN'

def partitions(length,seed):
    order=list(range(length)); random.Random(seed).shuffle(order)
    cut=length-4
    return {i:'train' if rank<cut else 'validation' if rank<cut+2 else 'test' for rank,i in enumerate(order)}

def make_record(identifier,text,split,families,intent=None,friction=None,**extra):
    return {'id':identifier,'text':text,'language':observed_language(text),'split':split,'group_id':families[0], 'family_ids':families,'intent':intent,'friction':friction,'reviewed':False,**extra}

def synthetic(bank):
    rows=[]; friction_base=collections.defaultdict(list)
    no_obstacle={'ADD_TASK','ORGANIZE_DAY','PRIORITIZE','TIME_ESTIMATION','RESCHEDULE','TASK_COMPLETED'}
    for label,pairs in bank['intent'].items():
        splits=partitions(len(pairs),101+list(bank['intent']).index(label))
        for index,pair in enumerate(pairs):
            family=f'synthetic-intent-{label}-{index:02}'
            for lang,template in enumerate(pair):
                objects=OBJECTS if '{task}' in template else [None]
                for object_index,obj in enumerate(objects):
                    # Actions use infinitive/first-person forms; other contexts use
                    # verbal nouns. These remain synthetic labels, not reviewed data.
                    if obj:
                        action=obj[0 if lang==0 else 2]; noun=obj[1 if lang==0 else 3]
                        value=action if label=='ADD_TASK' and index not in (8,) else noun
                        text=template.format(task=value)
                    else: text=template
                    friction=['UNKNOWN'] if label in no_obstacle else None
                    if label=='OVERWHELMED_WITH_TASKS': friction=['OVERWHELM']
                    if label=='PROCRASTINATION': friction=['AVOIDANCE']
                    if label=='INTERRUPTION_RECOVERY': friction=['INTERRUPTION']
                    row=make_record(f'{family}-{lang}-{object_index}',text,splits[index],[family],label,friction,source='SYNTHETIC',label_provenance='AI_AUTHORED_TEMPLATE',human_reviewed=False)
                    if obj:
                        start=text.index(value)
                        row['task_mentions']=[{'text':value,'start':start,'end':start+len(value)}]
                    rows.append(row)
    for label,pairs in bank['friction'].items():
        splits=partitions(len(pairs),301+list(bank['friction']).index(label))
        for index,pair in enumerate(pairs):
            family=f'synthetic-friction-{label}-{index:02}'
            friction_base[splits[index]].append((label,family,pair))
            for lang,text in enumerate(pair):
                # Preserve labels across a few neutral framing variants, all kept
                # within one family/split. More rows do not mean more independence.
                variants=[text,('Right now, ' if lang==0 else 'حاليا، ')+text,('About my work: ' if lang==0 else 'بالنسبة لعملي: ')+text]
                for variant,value in enumerate(variants):
                    rows.append(make_record(f'{family}-{lang}-{variant}',value,splits[index],[family],friction=[label],source='SYNTHETIC',label_provenance='AI_AUTHORED_TEMPLATE',human_reviewed=False))
    # Multi-label examples reuse only families already assigned to the same split.
    # Component family IDs remain available for leakage auditing.
    for split,choices in friction_base.items():
        rng=random.Random(700+PRIORITY[split])
        pairs=[(a,b) for i,a in enumerate(choices) for b in choices[i+1:] if a[0]!=b[0]]
        rng.shuffle(pairs)
        for index,(a,b) in enumerate(pairs[:90]):
            for lang in (0,1):
                text=a[2][lang]+' '+b[2][lang]
                rows.append(make_record(f'synthetic-multi-{split}-{index}-{lang}',text,split,[a[1],b[1]],friction=sorted([a[0],b[0]]),source='SYNTHETIC',label_provenance='AI_AUTHORED_COMPOSITION',human_reviewed=False))
    return rows

def public_rows():
    rows=[]
    # Read calendar intent means inspecting the calendar/agenda, not assigning
    # priorities. No public dataset row is used to invent a friction label.
    mapping={'calendar_set':'ADD_TASK','calendar_query':'ORGANIZE_DAY'}
    negatives={'weather_query','music_query','music_play','general_quirky','qa_factoid','qa_definition','qa_currency','iot_hue_lightoff','iot_hue_lighton','play_radio','calendar_remove'}
    selected_negative_ids=set()
    en=[json.loads(line) for line in (ROOT/'data/raw/massive/en-US.jsonl').read_text(encoding='utf-8').splitlines()]
    for partition in ('train','dev','test'):
        candidates=[r for r in en if r['partition']==partition and r['intent'] in negatives]
        candidates.sort(key=lambda r:hashlib.sha256(('negative-42-'+str(r['id'])).encode()).hexdigest())
        selected_negative_ids.update(str(r['id']) for r in candidates[:450 if partition=='train' else 120])
    for locale in ('en-US','ar-SA'):
        source=ROOT/f'data/raw/massive/{locale}.jsonl'
        for line in source.read_text(encoding='utf-8').splitlines():
            r=json.loads(line)
            if r['intent'] not in mapping and str(r['id']) not in selected_negative_ids: continue
            split={'train':'train','dev':'validation','test':'test'}[r['partition']]
            family='massive-'+str(r['id'])
            rows.append(make_record(f'{family}-{locale}',r['utt'],split,[family],mapping.get(r['intent'],'UNKNOWN'),None,source='MASSIVE_1.1',source_locale=locale,source_id=str(r['id']),source_intent=r['intent'],source_partition=r['partition'],source_annotated_utterance=r['annot_utt'],label_provenance='PUBLISHER_LABEL_WITH_DECLARED_MAPPING',human_reviewed=False,license='CC-BY-4.0'))
    return rows

def training_extra():
    bank=json.loads((ROOT/'data/planning/training_extra.json').read_text(encoding='utf-8'))
    rows=[]
    for task in ('intent','friction'):
        for label,pairs in bank[task].items():
            for index,pair in enumerate(pairs):
                family=f'synthetic-extra-{task}-{label}-{index:02}'
                for lang,template in enumerate(pair):
                    objects=OBJECTS if '{task}' in template else [None]
                    for object_index,obj in enumerate(objects):
                        text=template.format(task=obj[1 if lang==0 else 3]) if obj else template
                        variants=[text] if task=='intent' else [text,('Right now, ' if lang==0 else 'حاليا، ')+text,('About my work: ' if lang==0 else 'بالنسبة لعملي: ')+text]
                        for v,value in enumerate(variants):
                            rows.append(make_record(f'{family}-{lang}-{object_index}-{v}',value,'train',[family],intent=label if task=='intent' else None,friction=[label] if task=='friction' else None,source='SYNTHETIC',label_provenance='AI_AUTHORED_TRAINING_EXPANSION',human_reviewed=False))
    return rows

def write(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')

def main():
    bank=json.loads((ROOT/'data/planning/seed_families.json').read_text(encoding='utf-8'))
    rows=synthetic(bank)+training_extra()+public_rows()
    # Exact duplicate text must never cross splits. Keep the latest holdout and
    # remove entire affected lower-priority families; never move public test to train.
    by_text=collections.defaultdict(list)
    for row in rows: by_text[normalize(row['text'])].append(row)
    removed_families=set()
    for group in by_text.values():
        top=max(PRIORITY[r['split']] for r in group)
        for row in group:
            if PRIORITY[row['split']]<top: removed_families.update(row['family_ids'])
    # Unlabeled lexical decontamination. Translations already share a family.
    # Drop lower-priority whole families, never reassign official public test rows.
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors
    unique_texts=sorted(by_text)
    audit_features=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),max_features=30000).fit_transform(unique_texts)
    neighbors=NearestNeighbors(n_neighbors=min(5,len(unique_texts)),metric='cosine',n_jobs=1).fit(audit_features)
    near_pairs=set()
    for start in range(0,len(unique_texts),256):
        distances,indices=neighbors.kneighbors(audit_features[start:start+256])
        for offset,(ds,js) in enumerate(zip(distances,indices)):
            i=start+offset
            for distance,j in zip(ds,js):
                if i==j or distance>.04: continue
                a,b=by_text[unique_texts[i]],by_text[unique_texts[int(j)]]
                if len({r['split'] for r in a+b})<=1: continue
                near_pairs.add(tuple(sorted((i,int(j)))))
                top=max(PRIORITY[r['split']] for r in a+b)
                for r in a+b:
                    if PRIORITY[r['split']]<top: removed_families.update(r['family_ids'])
    quarantined=[r for r in rows if any(f in removed_families for f in r['family_ids'])]
    rows=[r for r in rows if not any(f in removed_families for f in r['family_ids'])]
    unique={}
    for row in rows:
        key=(row['split'],normalize(row['text']),row['intent'],tuple(row['friction'] or []))
        if key not in unique: unique[key]=row
    rows=sorted(unique.values(),key=lambda r:r['id'])
    family_splits=collections.defaultdict(set)
    for row in rows:
        for family in row['family_ids']: family_splits[family].add(row['split'])
    assert all(len(s)==1 for s in family_splits.values()),'Family leakage'
    for split in PRIORITY: write(ROOT/f'data/planning/{split}.jsonl',[r for r in rows if r['split']==split])
    write(ROOT/'data/planning/all.jsonl',rows)
    write(ROOT/'data/planning/quarantine.jsonl',quarantined)
    with (ROOT/'data/planning/planning_dataset.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        fields=['id','text','language','source','split','group_id','intent','friction','human_reviewed']
        writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader()
        for row in rows:
            record={key:row.get(key) for key in fields}
            record['friction']=json.dumps(row['friction'],ensure_ascii=False) if row['friction'] is not None else ''
            writer.writerow(record)
    summary={'seed':42,'records':len(rows),'bySource':dict(collections.Counter(r['source'] for r in rows)),'byLanguage':dict(collections.Counter(r['language'] for r in rows)),'bySplit':dict(collections.Counter(r['split'] for r in rows)),'intentLabeled':sum(r['intent'] is not None for r in rows),'frictionLabeled':sum(r['friction'] is not None for r in rows),'templateOrPublicUtteranceFamilies':len(family_splits),'quarantinedCrossSplitDuplicateRows':len(quarantined),'nearDuplicateAudit':{'similarityMinimum':.96,'nearestNeighbors':5,'crossSplitPairsBeforeQuarantine':len(near_pairs),'method':'Character TF-IDF; remove entire lower-priority families; audit uses text only, never test labels. Not proof of semantic independence.'},'familyLeakage':False,'humanReviewed':False,'publicIntentMapping':{'calendar_set':'ADD_TASK','calendar_query':'ORGANIZE_DAY','selected_nonplanning_intents':'UNKNOWN'},'noFrictionLabelsGuessedForMassive':True,'warning':'Synthetic holdouts measure template-family generalization only; not patient or clinical validation. Calendar mapping is coarse and does not imply full slot/date understanding.'}
    (ROOT/'reports/planning-data.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
