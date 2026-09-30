import re, unicodedata

NORMALIZER_VERSION='en-ar-1.0.0'
def normalize(text):
    text=unicodedata.normalize('NFKC',text).lower().replace('’',"'")
    text=re.sub(r'[\u064b-\u065f\u0670\u0640]','',text)
    text=text.translate(str.maketrans('أإآٱى٠١٢٣٤٥٦٧٨٩','ااااي0123456789'))
    return re.sub(r'\s+',' ',text).strip()

def languages(text):
    return [label for label,pattern in [('EN','[A-Za-z]'),('AR','[\u0600-\u06ff]')] if re.search(pattern,text)]

def choose(lang,en,ar): return ar if lang=='AR' else en

STOPWORDS={'the','a','an','my','our','your','his','her','their','to','for','of','and','at','on','in','up','with','this','that','it','some','any','from','into','ال','الى','من','في','على','مع','هذا','هذه','عن'}
# Leading verbs name the action; what identifies the task to the user is its object.
LEAD_VERBS={'finish','complete','do','make','get','take','call','email','send','buy','pay','write','read','study','review','book','submit','open','clean','wash','fix','print','order','renew','refill','collect','attend','visit','meet','water','pick','check','اخلص','انهي','اكلم','اتصل','ارسل','اشتري','اشتر','ادفع','اكتب','اقرا','ادرس','اراجع','احجز','افتح','انظف','اغسل','اخذ','اقابل','ازور','اجدد','اطبع','اطلب','اجمع'}

def _stem(word):
    for suffix in ('ing','ed','es','s'):
        if len(word)>len(suffix)+2 and word.endswith(suffix): return word[:-len(suffix)]
    return word

def mentions(title,text):
    """True when the message names this task by its object, allowing inflection.

    A stored "finish the report" is recognized in "I finished the report" without
    matching it against an unrelated task. Returns False when the title carries no
    identifying object, so a bare "I'm done" never closes a guessed commitment.
    """
    words=[w for w in re.findall(r'[\w؀-ۿ]+',normalize(title))]
    while words and words[0] in LEAD_VERBS: words=words[1:]
    content={_stem(w) for w in words if w not in STOPWORDS and len(w)>=3}
    if not content: return False
    said={_stem(w) for w in re.findall(r'[\w؀-ۿ]+',normalize(text))}
    return content<=said
