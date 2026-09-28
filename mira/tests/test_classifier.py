from vitamind.ml.classifier import MiraMLClassifier


def test_classifier_loads_artifact_lazily_and_returns_contract():
    result = MiraMLClassifier().predict("I have been distracted since school.")
    assert result.routing_label in {"ADHD", "AMBIGUOUS", "BIPOLAR", "HEALTHY", "OTHER", "PSYCHOSIS"}
    assert 0 <= result.top_score_internal <= 1
    assert result.top_score_internal >= result.second_score_internal