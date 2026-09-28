"""Read native patient text for follow-up selection, not symptom confirmation."""
from .classifier import MiraMLClassifier
from ..clinical.feature_extractor import RULES
from .bilingual_text import INPUT_TYPE, build_input

LABELS = {'ADHD': 'ADHD', 'BIPOLAR': 'BIPOLAR_SPECTRUM', 'PSYCHOSIS': 'PSYCHOSIS_SPECTRUM'}

class IntegratedModel:
    def __init__(self, classifier=None):
        self.classifier = classifier or MiraMLClassifier()

    def assess(self, state, text=None):
        base = {'clinical_validation': False, 'features_used': [],
                'used_for': 'followup_priority_and_concordance', 'model_input': '',
                'input_type': INPUT_TYPE, 'native_arabic_model': False}
        try:
            # Explicitly injected capture classifiers may declare their input
            # contract. The persisted artifact always supplies its own metadata.
            info = self.classifier.model_info() if hasattr(self.classifier, 'model_info') else {
                'input_type': getattr(self.classifier, 'input_type', INPUT_TYPE), 'languages': ['en', 'ar']}
            mode = info.get('input_type', 'structured_evidence_projection')
            if mode == INPUT_TYPE:
                model_text, used = build_input(state, text)
            else:
                used = [r.feature for r in RULES if state.get_status(r.domain, r.feature).value == 'present']
                model_text = ' '.join('I ' + r.present[0] + '.' for r in RULES if r.feature in used)
            base.update(input_type=mode, model_input=model_text, features_used=used,
                        native_arabic_model=mode == INPUT_TYPE and 'ar' in info.get('languages', []),
                        trained_languages=info.get('languages', ['en']))
            if not model_text:
                return dict(base, status='insufficient_evidence', suggested_condition=None)
            result = self.classifier.predict(base['model_input'])
            # Engineering guard only; never describe this threshold as clinical validity.
            confident = result.top_score_internal >= .45 and result.margin_internal >= .10
            return dict(base, status='available', routing_label=result.routing_label,
                        suggested_condition=LABELS.get(result.routing_label) if confident else None,
                        unconfirmed_followup_only=mode == INPUT_TYPE and confident and not used,
                        scores=result.class_scores_internal, margin=result.margin_internal,
                        abstained=not confident or result.routing_label not in LABELS,
                        legacy_safety_score_used=False)
        except Exception as exc:
            return dict(base, status='unavailable', suggested_condition=None, error_type=type(exc).__name__)
