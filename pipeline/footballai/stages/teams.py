import numpy as np
import pandas as pd
import umap
from sklearn.cluster import KMeans
from footballai.config import Config
from footballai import schema


def cluster_embeddings(embeddings: np.ndarray, cfg: Config) -> np.ndarray:
    n_comp = min(cfg.teams.get("umap_components", 3), embeddings.shape[1], len(embeddings) - 1)
    reducer = umap.UMAP(n_components=n_comp, random_state=cfg.seed)
    reduced = reducer.fit_transform(embeddings)
    km = KMeans(n_clusters=2, random_state=cfg.seed, n_init=10)
    return km.fit_predict(reduced)


def _embed_crops(crops: list[np.ndarray], cfg: Config) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoProcessor
    from PIL import Image
    name = cfg.teams["siglip_model"]
    processor = AutoProcessor.from_pretrained(name)
    model = AutoModel.from_pretrained(name).eval()
    imgs = [Image.fromarray(c[:, :, ::-1]) for c in crops]  # BGR->RGB
    with torch.no_grad():
        inputs = processor(images=imgs, return_tensors="pt")
        feats = model.get_image_features(**inputs)
    return feats.cpu().numpy()


def assign_teams(tracks: pd.DataFrame, video_path: str, cfg: Config) -> pd.DataFrame:
    from footballai.stages.extract import iter_frames
    out = tracks.copy()
    out["team"] = None
    out.loc[out["class"] == "goalkeeper", "team"] = "GK"
    out.loc[out["class"] == "referee", "team"] = "referee"

    players = out[out["class"] == "player"]
    frames_needed = set(players["frame"])
    crops, index = [], []
    for fidx, frame in iter_frames(video_path):
        if fidx not in frames_needed:
            continue
        for ridx, r in players[players["frame"] == fidx].iterrows():
            x, y, w, h = int(r.bbox_x), int(r.bbox_y), int(r.bbox_w), int(r.bbox_h)
            crop = frame[max(0, y):y + h, max(0, x):x + w]
            if crop.size == 0:
                continue
            crops.append(crop)
            index.append(ridx)
    if crops:
        emb = _embed_crops(crops, cfg)
        labels = cluster_embeddings(emb, cfg)
        mapping = {0: "A", 1: "B"}
        for ridx, lab in zip(index, labels):
            out.at[ridx, "team"] = mapping[int(lab)]
    out = out[schema.TEAM_COLUMNS]
    schema.validate(out, schema.TEAM_COLUMNS)
    return out
