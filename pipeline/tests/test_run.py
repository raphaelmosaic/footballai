from footballai.run import resolve_stage_range, STAGES

def test_full_range_default():
    assert resolve_stage_range(None, None) == STAGES

def test_subrange_inclusive():
    assert resolve_stage_range("track", "project") == ["track", "teams", "project"]

def test_invalid_stage_raises():
    import pytest
    with pytest.raises(ValueError):
        resolve_stage_range("nope", None)
