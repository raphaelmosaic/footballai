import pandas as pd
from footballai import schema
from footballai.stages.export import write_artifacts, read_tracks

def test_write_and_read_round_trip(tmp_path):
    df = pd.DataFrame({c: [0] for c in schema.FINAL_COLUMNS})
    write_artifacts(df, {"fps": 25.0}, str(tmp_path))
    assert (tmp_path / "tracks.parquet").exists()
    assert (tmp_path / "tracks.json").exists()
    assert (tmp_path / "meta.json").exists()
    back = read_tracks(str(tmp_path))
    assert list(back.columns) == schema.FINAL_COLUMNS
