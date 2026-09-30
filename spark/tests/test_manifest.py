"""MANIFEST.json pins the datasets and trained models. When they change on purpose:
    python -m training.build_manifest
`python -m training.build_manifest --check` verifies the whole tree; only data/ and models/
are enforced here, so editing source files never breaks the suite.
The manifest went stale once (45 changed files, 13 pointing at a moved package).
"""
from training.build_manifest import problems


def test_manifest_matches_the_shipped_files():
    assert problems(('data/', 'models/')) == [], "run `python -m training.build_manifest` after changing shipped files"
