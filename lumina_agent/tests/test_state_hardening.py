"""Garbage in a check-in is absence, never an extreme reading."""
import pytest

from lumina.state import build_state


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), "abc", True, [], {}])
def test_unusable_values_are_missing_not_clamped(bad):
    state = build_state({"stress": bad, "sleep_hours": bad})
    assert state.quality("stress") == "missing" and state.value("stress") is None
    assert state.quality("sleep") == "missing" and state.value("sleep") is None


def test_out_of_range_numbers_still_clamp_and_numeric_strings_parse():
    state = build_state({"energy": 99, "focus": "4", "mood": -3})
    assert (state.value("energy"), state.value("focus"), state.value("mood")) == (10.0, 4.0, 0.0)


def test_journal_signals_get_the_same_treatment():
    state = build_state({}, {"stress": float("nan"), "mood": 6})
    assert state.value("stress") is None and state.value("mood") == 6.0
