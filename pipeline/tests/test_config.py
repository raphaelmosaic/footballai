from footballai.config import load_config

def test_load_config_reads_pitch_and_seed(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        "seed: 42\n"
        "pitch_length: 105.0\n"
        "pitch_width: 68.0\n"
        "paths: {work_dir: work}\n"
        "detect: {weights: w.pt, conf: 0.25}\n"
        "track: {}\n"
        "teams: {}\n"
        "refine: {}\n"
    )
    cfg = load_config(str(cfg_file))
    assert cfg.seed == 42
    assert cfg.pitch_length == 105.0
    assert cfg.detect["conf"] == 0.25
