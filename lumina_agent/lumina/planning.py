"""Putting a patient's own list in order, in their own words.

"Help me prioritize my day" used to be answered from the check-in history - an
overload concern turned it into a canned exercise that ignored what was asked - and a
list of things to do was read as small talk about friends. A request to plan is now its
own conversational move (`PLAN_REQUEST` in intent.py), and this module does the small
amount of work it needs: pull the items out of the message and order them.

The order is deliberately plain and explainable, and it claims nothing about what
matters to the patient:

* things with a time that will not move (an appointment, a prayer, a class) first;
* things the patient called urgent, or gave a deadline, next;
* everything else in the order they wrote it;
* a social plan last, because it is the one most easily moved.

The reply says the order rests on names alone, so the patient corrects it rather than
trusting it. Items are the patient's own words, cut down to a short phrase; nothing is
inferred, translated or invented, and an item that trips a content rule is left out
instead of being echoed.
"""
from __future__ import annotations

import re

from .text import normalize

MAX_ITEMS = 8
MAX_ITEM_CHARS = 60

# The request half of a message ("... so help me organise them by priority") is an
# instruction, not an item. Cut it off before the list is read.
REQUEST_TAIL = re.compile(
    r"[,;.\s]*(?:\b(?:so|and|then|now|please)\s+)*(?:(?:can|could|would|will) you\s+)?"
    r"(?:help(?:\s+me)?(?:\s+to)?\b|(?:organi[sz]e|prioriti[sz]e|plan|sort|rank|order)\s+"
    r"(?:them|these|those|it|all|my|the day|today|everything|things)\b"
    r"|و?(?:ساعدني|رتبهم|رتبها|رتب لي|رتب يومي|رتب|حدد الاولويات)).*$", re.I)

# The request can also come first: "Help me prioritize my day: doctor at 4pm, study". The
# head is cut off only when it names the day/list itself, so "order pizza" stays an item.
REQUEST_HEAD = re.compile(
    r"^(?:(?:please|hey|hi)[,\s]+)*(?:(?:can|could|would|will) you\s+)?"
    r"(?:help\b(?:\s+me\b)?(?:\s+to\b)?\s*)?"
    r"(?:(?:organi[sz]e|prioriti[sz]e|plan|sort|rank|order)\s+(?:my|the|these|those|them|it|all|today|everything)"
    r"(?:\s+(?:day|tasks?|things|list|schedule|priorities|to-?do))*)?\s*[:;,–—-]\s*"
    r"|^(?:ساعدني|رتب لي|رتب)[^:;،,]{0,30}[:،,]\s*", re.I)

# Lead-ins that introduce an intention; stripped so the item is just the thing itself.
LEAD_IN = re.compile(
    r"^(?:(?:and|then|also|first|next|later|finally|after that|so)\s+)*(?:please\s+)?"
    r"(?:i(?:'m| am)\s+(?:going|gonna)\s+to|i(?:'ll| will)|i plan to|i(?:'d| would) like to|"
    r"i\s+(?:need|have|want|ought|got|must|should)\s+to|i\s+gotta|i(?:'ve)?\s+got\s+to|"
    r"لازم|احتاج(?: ان)?|ابغي|اريد(?: ان)?|عندي|يجب(?: ان)?)\s+", re.I)
# "I am going to the doctor" names a place, not a verb: it means "go to the doctor".
DESTINATION = re.compile(
    r"^(?:(?:and|then|also|first|next|later|finally|so)\s+)*(?:i(?:'m| am)\s+)?"
    r"(?:going|heading|off)\s+to\s+((?:the|my|a|an)\s+.+)$", re.I)

# Verbs that start a new item after "and": "see the dentist and call mom" is two things,
# "salt and pepper" is one.
ACTION_VERBS = (r"(?:call|phone|text|email|send|buy|get|pick|pay|study|revise|read|review|write|"
                r"finish|clean|wash|cook|see|meet|visit|go|pray|exercise|walk|book|submit|prepare|"
                r"fix|print|take|do|rest|sleep|nap|shower|eat|reply|check|pack|tidy)")
SPLIT = re.compile(
    r"[,;\n،؛]+"
    r"|\s+and\s+(?=i\b|we\b|(?:i|we)'|going\b|heading\b|to\b|then\b|also\b)"
    r"|\s+(?:and\s+)?(?:then|also|after that)\s+"
    r"|\s+and\s+(?=" + ACTION_VERBS + r"\b)"
    r"|\s+و(?=[ا-ي]{2,})", re.I)

FIXED_TIME = re.compile(
    r"\b(?:appointment|doctor|dentist|clinic|hospital|therapist|psychiatrist|checkup|check-up|"
    r"meeting|interview|exam|class|lecture|flight|train|shift|pray|prayer|salah|salat)\b"
    r"|موعد|طبيب|دكتور|عياد[هة]|مستشفى|اجتماع|مقابل[هة]|امتحان|محاضر[هة]|حص[هة]|صلا[هة]|اصلي|صلي")
PREPARING = re.compile(r"^(?:study|revise|prepare|read|review|اذاكر|ادرس|اراجع)\b")
URGENT = re.compile(r"\b(?:urgent|asap|deadline|due|overdue)\b|مستعجل|ضروري")
SOCIAL = re.compile(
    r"\b(?:friends?|coffee|hang out|catch up|party|movie|game|gaming|shopping)\b"
    r"|صديق|اصدقاء|قهو[هة]|سهر[هة]|فيلم")
CLOCK = re.compile(r"\b(?:at\s+)?\d{1,2}(?::\d{2})?\s*(?:am|pm)\b|\bat\s+\d{1,2}(?::\d{2})\b|الساع[هة]\s+\d")
DOTS = re.compile(r"[\s.!?؟,;:،؛-]+$")


def extract_items(text):
    """The things the patient listed, as short phrases in their own words."""
    raw = re.sub(r"\s+", " ", text or "").strip()
    head = REQUEST_HEAD.match(raw)
    if head and raw[head.end():].strip():
        raw = raw[head.end():].strip()
    tail = REQUEST_TAIL.search(raw)
    if tail and raw[:tail.start()].strip():
        raw = raw[:tail.start()].strip()
    items = []
    for segment in SPLIT.split(raw):
        segment = (segment or "").strip()
        if not segment:
            continue
        destination = DESTINATION.match(segment)
        item = "go to " + destination[1].strip() if destination else LEAD_IN.sub("", segment).strip()
        item = DOTS.sub("", item)
        if len(item) < 3 or item.lower() in ("it", "them", "that", "this", "things", "stuff"):
            continue
        items.append(item[:MAX_ITEM_CHARS].rstrip())
    return items[:MAX_ITEMS]


def _score(item):
    name = normalize(item)
    score = 0
    if FIXED_TIME.search(name) and not PREPARING.match(name):
        score += 3
    if CLOCK.search(name):
        score += 1
    if URGENT.search(name):
        score += 2
    if SOCIAL.search(name) and not CLOCK.search(name):
        score -= 1
    return score


def order_items(items):
    """Stable order: fixed-time and urgent first, social last, otherwise as written.

    Returns (ordered, from_names_only). `from_names_only` is False when a stated clock
    time or an urgency word separated the items, so the reply only asks for corrections
    when the order really rests on the names alone.
    """
    scores = [_score(item) for item in items]
    ordered = [item for _, _, item in sorted(
        ((-score, index, item) for index, (score, item) in enumerate(zip(scores, items))))]
    stated = any(CLOCK.search(normalize(item)) or URGENT.search(normalize(item)) for item in items)
    return ordered, len(set(scores)) > 1 and not stated


def shown(item):
    return item[:1].upper() + item[1:]
