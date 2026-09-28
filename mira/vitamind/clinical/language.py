"""Shared bounded normalization; not a general language understanding model."""
import re
import unicodedata

def normalize(text):
    text = unicodedata.normalize('NFKC', text).lower().replace('’', "'").replace('‘', "'")
    text = re.sub(r'[\u064b-\u065f\u0670\u0640]', '', text)
    text = text.translate(str.maketrans('أإآىؤئ', 'ااايوي'))
    return re.sub(r'\s+', ' ', text).strip()

THIRD = re.compile(r'\b(?:my (?:brother|sister|friend|mother|father|patient|partner|son|daughter)|he|she|they)\b|(?:اخي|اختي|صديقي|صديقتي|والدي|والدتي|ابني|ابنتي)')
SELF = re.compile(r'\b(?:i|me|myself)\b|(?:انا|نفسي)')
EDUCATIONAL = re.compile(r'\b(?:essay|article|research|novel|movie|fiction|roleplay|example sentence|quote)\b|(?:مقال|بحث|رواية|فيلم|مثال لغوي)')
NEGATION = re.compile(r"\b(?:not|never|no|without|cannot|can't|don't|dont|doesn't|didn't|didnt|isn't|wasn't|wouldn't)\b|(?<!\w)(?:لا|لم|لن|ليس|لست|بدون|بلا)(?!\w)")

def clauses(text):
    return [p.strip() for p in re.split(r'[.!?؟,،;؛\n]+|\b(?:but|however)\b|\band (?=i\b)|(?:ولكن|لكن|بل)(?=\s)', normalize(text)) if p.strip()]

def patient_clauses(text):
    other = False
    educational = False
    for clause in clauses(text):
        if THIRD.search(clause) and not SELF.search(clause): other = True
        elif SELF.search(clause): other = False
        if EDUCATIONAL.search(clause): educational = True
        elif SELF.search(clause): educational = False
        yield clause, not other and not educational

def negated_before(clause, index):
    prefix = re.sub(r'\bnot only\b', '', clause[:index])
    return bool(NEGATION.search(' '.join(prefix.split()[-7:])))

def occurrences(clause, phrase):
    normalized = normalize(phrase)
    prefix = '[وف]?' if any('\u0600' <= c <= '\u06ff' for c in normalized) else ''
    return re.finditer(r'(?<!\w)' + prefix + re.escape(normalized) + r'(?!\w)', clause)
