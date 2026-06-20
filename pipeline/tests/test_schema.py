import pandas as pd
import pytest
from footballai import schema

def test_final_columns_superset_of_detection():
    assert set(schema.DETECTION_COLUMNS).issubset(set(schema.FINAL_COLUMNS))

def test_validate_raises_on_missing_columns():
    df = pd.DataFrame({"frame": [0]})
    with pytest.raises(ValueError, match="missing columns"):
        schema.validate(df, schema.DETECTION_COLUMNS)

def test_validate_passes_when_columns_present():
    df = pd.DataFrame({c: [] for c in schema.DETECTION_COLUMNS})
    schema.validate(df, schema.DETECTION_COLUMNS)  # no raise
