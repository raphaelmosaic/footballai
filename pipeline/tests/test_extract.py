import pytest
from tests.fixtures.make_clip import make_clip
from footballai.pipeline.extract import read_meta, iter_frames

def test_read_meta_raises_on_bad_path():
    with pytest.raises(FileNotFoundError):
        read_meta("/nonexistent/nope.mp4")

def test_iter_frames_raises_on_bad_path():
    with pytest.raises(FileNotFoundError):
        list(iter_frames("/nonexistent/nope.mp4"))

def test_read_meta(tmp_path):
    clip = str(tmp_path / "c.mp4")
    make_clip(clip, n=10, w=64, h=48, fps=5)
    meta = read_meta(clip)
    assert meta.width == 64 and meta.height == 48
    assert meta.fps == 5.0
    assert meta.n_frames == 10

def test_iter_frames_yields_indexed_frames(tmp_path):
    clip = str(tmp_path / "c.mp4")
    make_clip(clip, n=6)
    frames = list(iter_frames(clip))
    assert [i for i, _ in frames] == [0, 1, 2, 3, 4, 5]
    assert frames[0][1].shape[2] == 3
