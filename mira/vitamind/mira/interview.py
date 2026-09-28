"""Deterministic missing-information planner with learned-model tie breaking."""
from .questions import BY_KEY, QUESTIONS, Question
from ..clinical.feature_extractor import RULES

DEFAULT_SESSION_QUESTIONS = 10

class InterviewPlanner:
    def plan(self, state, answers, model, skipped=()):
        # Conflicting observations must be clarified before increasing confidence.
        for rule in RULES:
            key = 'clarify:' + rule.feature
            if state.get_status(rule.domain,rule.feature).value == 'conflicting' and key not in skipped:
                known = next((q for q in QUESTIONS if q.target == (rule.domain,rule.feature)),None)
                return Question(key,
                    'I have conflicting answers about this. Which is accurate now? ' + (known.en if known else 'Do you currently experience '+rule.feature.replace('_',' ')+'?'),
                    'لدي إجابات متعارضة حول هذه النقطة. أيها يصف حالتك الآن؟ ' + (known.ar if known else 'هل ما ذكرته سابقاً عن هذا العرض ينطبق عليك الآن؟'),
                    (rule.domain,rule.feature), 'clarify', 'CLARIFICATION')
        if 'concern' not in answers and 'concern' not in skipped:
            return BY_KEY['concern']
        counts = {name:sum(state.get_status(r.domain,r.feature).value=='present' for r in RULES if
                  r.domain in domains) for name,domains in {
            'ADHD':('attention','developmental_history'),
            'BIPOLAR_SPECTRUM':('mood','sleep','episode_history'),
            'PSYCHOSIS_SPECTRUM':('psychosis',)}.items()}
        preferred = model.get('suggested_condition')
        # The model can change follow-up order, never insert a symptom or diagnosis.
        order = sorted(counts,key=lambda k:(-counts[k], -(k==preferred)))
        if preferred in counts and (counts[preferred]>0 or model.get('unconfirmed_followup_only')):
            order.remove(preferred)
            order.insert(0,preferred)
        core = ['duration','impairment','confounders']
        branches = {
            'ADHD':['attention','childhood','settings'],
            'BIPOLAR_SPECTRUM':['sleep_need','energy','episodes'],
            'PSYCHOSIS_SPECTRUM':['voices','beliefs'],
        }
        # Ask core context early, then the most supported branch, then brief coverage.
        keys = [key for group in order for key in branches[group]]
        keys = keys[:2] + core + keys[2:]
        for key in keys:
            q=BY_KEY[key]
            if key in answers or key in skipped: continue
            if q.target and state.get_status(*q.target).value in {'present','absent'}: continue
            # No childhood/settings interrogation without an attention signal.
            if key in {'childhood','settings'} and counts['ADHD']==0: continue
            if key=='episodes' and counts['BIPOLAR_SPECTRUM']==0: continue
            return q
        return None
