"""Independent safety guard. Bounded rules, not a validated crisis classifier."""
import re
from dataclasses import dataclass
from ..clinical.language import normalize, patient_clauses, occurrences, negated_before

@dataclass
class SafetyResult:
    urgent: bool
    reason: str | None = None
    needs_clarification: bool = False
    context: str | None = None

class SafetyDetector:
    PATTERNS = (
        ('suicidal_intent', ('kill myself', 'suicide', 'end my life', 'hurt myself', 'want to die', 'take my own life',
         'اقتل نفسي', 'ساقتل نفسي', 'انتحار', 'الانتحار', 'انتحر', 'سانتحر', 'انهي حياتي', 'سانهي حياتي', 'انهاء حياتي', 'اريد ان اموت', 'اوذي نفسي')),
        ('homicidal_intent', ('kill someone', 'hurt someone', 'harm others', 'اقتل شخصا', 'اقتل احدا', 'اوذي شخصا', 'اوذي احدا', 'الحق الاذي بالاخرين')),
        ('immediate_danger', ('in immediate danger', 'not safe right now', 'في خطر فوري', 'في خطر مباشر', 'لست امنا الان', 'لست بامان', 'لست في امان')),
        ('command_hallucination', ('voices tell me to jump', 'voices telling me to jump', 'voices order me to attack', 'اصوات تامرني بالقفز', 'اصوات تامرني بالهجوم')),
    )
    VAGUE = ('cannot go on', "can't go on", 'better off without me', 'disappear forever', "don't want to live", 'wish i were dead', 'لا استطيع الاستمرار', 'اتمني ان اختفي', 'افضل بدوني', 'لا اريد العيش')
    HISTORICAL = re.compile(r'\b(?:years ago|last year|used to|in the past)\b|(?:منذ سنوات|العام الماضي|في الماضي)')
    NOW = re.compile(r'\b(?:now|tonight|today|will|going to)\b|(?:الان|الليله|الليلة|اليوم|سوف|ساقتل|سانهي)')

    def detect(self, state, text):
        clarification = False
        context = None
        for clause, own in patient_clauses(text):
            for feature, phrases in self.PATTERNS:
                for phrase in phrases:
                    for match in occurrences(clause, phrase):
                        if not own:
                            context = 'other_person_or_educational'
                            continue
                        inability_to_stop = re.search(r"(?:can't|cannot) (?:stop|resist)|لا استطيع التوقف",clause[:match.start()])
                        if negated_before(clause, match.start()) and not inability_to_stop: continue
                        if self.HISTORICAL.search(clause) and not self.NOW.search(clause):
                            clarification = True
                            context = 'historical_safety_concern'
                            continue
                        if not any(o.feature == feature and o.domain == 'safety' for o in state.observations):
                            state.add_observation('safety', feature, 'present', 1, text[:500], 'SAFETY')
                        return SafetyResult(True, feature)
            if own and any(normalize(p) in clause for p in self.VAGUE): clarification = True
        return SafetyResult(False, needs_clarification=clarification, context=context)
