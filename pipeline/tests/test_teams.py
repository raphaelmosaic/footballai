import numpy as np
from footballai.config import load_config
from footballai.pipeline.teams import cluster_embeddings

def test_cluster_separates_two_blobs():
    cfg = load_config()
    rng = np.random.default_rng(0)
    a = rng.normal(-5, 0.2, size=(20, 8))
    b = rng.normal(5, 0.2, size=(20, 8))
    emb = np.vstack([a, b])
    labels = cluster_embeddings(emb, cfg)
    assert set(labels) == {0, 1}
    # all of blob A share one label, all of blob B the other
    assert len(set(labels[:20])) == 1
    assert len(set(labels[20:])) == 1
    assert labels[0] != labels[20]
