"""PyTorch Dataset over StatsBomb event files.

Each sample is one event from one match, returned as (state, labels, mask).
Matches are processed lazily and cached with a small LRU cache so only a few
matches are held in RAM at a time.
"""

import json
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .state_builder import MatchState, load_events, load_lineups


def _manifest_path() -> Path:
    return Path(__file__).with_name(".statsbomb_dataset_index.json")


def _scan_matches(events_dir: Path, manifest_path: Path) -> List[Tuple[int, int]]:
    """Return [(match_id, n_events), ...] by scanning the events directory."""
    counts = []
    for path in sorted(events_dir.glob("*.json")):
        match_id = int(path.stem)
        # We only need the length of the event array; do not keep the events.
        with path.open(encoding="utf-8") as f:
            events = json.load(f)
        counts.append((match_id, len(events)))
    counts.sort(key=lambda x: x[0])
    manifest_path.write_text(json.dumps(counts), encoding="utf-8")
    return counts


def _load_or_build_index(
    data_root: Path, max_matches: Optional[int]
) -> List[Tuple[int, int]]:
    events_dir = data_root / "events"
    manifest = _manifest_path()
    n_event_files = len(list(events_dir.glob("*.json")))

    if manifest.exists():
        try:
            counts = json.loads(manifest.read_text(encoding="utf-8"))
            if len(counts) == n_event_files:
                counts = [(int(mid), int(n)) for mid, n in counts]
                if max_matches is not None:
                    counts = counts[:max_matches]
                return counts
        except Exception:
            pass

    if max_matches is None:
        counts = _scan_matches(events_dir, manifest)
        return counts

    # Fast path: only the requested number of matches are needed.
    paths = sorted(events_dir.glob("*.json"), key=lambda p: int(p.stem))[:max_matches]
    counts = []
    for path in paths:
        with path.open(encoding="utf-8") as f:
            events = json.load(f)
        counts.append((int(path.stem), len(events)))
    counts.sort(key=lambda x: x[0])
    return counts


class StatsBombDataset(Dataset):
    """PyTorch Dataset over StatsBomb open-data events.

    Args:
        data_root: Path to the directory containing 'events' and 'lineups' folders.
        horizon_seconds: Look-ahead window in seconds used for labels.
        max_matches: Optional cap on the number of matches to include.
        cache_matches: Number of fully-processed matches to keep in memory.

    Returns:
        state: torch.Tensor of shape (23, 9)
            row 0  : ball [x, y, 0, 0, 0, 0, 0, 0, 0]
            rows 1-22: players [x, y, vx, vy, team0, team1, pos_id, is_poss, on_pitch]
        labels: dict of torch.Tensor
            pass_receiver: (3,)  [team_rel_x, team_rel_y, receiver_slot]
                           receiver_slot == -1 if no pass within horizon.
            shot_score: scalar    statsbomb xG if a shot occurs within horizon.
            turnover: scalar      1 if possession changes within horizon.
        mask: torch.Tensor of shape (22,)  1 for on-pitch players, 0 otherwise.
    """

    def __init__(
        self,
        data_root: str = "/home/jack/workspace/open-data/data",
        horizon_seconds: float = 5.0,
        max_matches: Optional[int] = None,
        cache_matches: int = 4,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.events_dir = self.data_root / "events"
        self.lineups_dir = self.data_root / "lineups"
        self.horizon = horizon_seconds
        self.cache_matches = max(1, cache_matches)

        self._counts = _load_or_build_index(self.data_root, max_matches)
        self._match_ids = [mid for mid, _ in self._counts]
        self._cum_lengths = np.cumsum([n for _, n in self._counts]).tolist()
        self._cache: OrderedDict[int, Tuple[List[torch.Tensor], List[Dict[str, torch.Tensor]], List[torch.Tensor]]] = OrderedDict()

    def __len__(self) -> int:
        return int(self._cum_lengths[-1]) if self._cum_lengths else 0

    def _find_match(self, idx: int) -> Tuple[int, int]:
        """Map global index to (match_id, local_index)."""
        import bisect
        pos = bisect.bisect_left(self._cum_lengths, idx + 1)
        prev = self._cum_lengths[pos - 1] if pos > 0 else 0
        local_idx = idx - prev
        return self._match_ids[pos], local_idx

    def _load_match(self, match_id: int) -> Tuple[List[torch.Tensor], List[Dict[str, torch.Tensor]], List[torch.Tensor]]:
        if match_id in self._cache:
            self._cache.move_to_end(match_id)
            return self._cache[match_id]

        events = load_events(match_id, str(self.data_root))
        lineups = load_lineups(match_id, str(self.data_root))
        state = MatchState(events, lineups, horizon_seconds=self.horizon)
        states = [state[i][0] for i in range(len(state))]
        labels = [state[i][1] for i in range(len(state))]
        masks = [state[i][2] for i in range(len(state))]
        sample = (states, labels, masks)

        self._cache[match_id] = sample
        self._cache.move_to_end(match_id)
        while len(self._cache) > self.cache_matches:
            self._cache.popitem(last=False)
        return sample

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"index {idx} out of range for dataset of size {len(self)}")
        match_id, local_idx = self._find_match(idx)
        states, labels, masks = self._load_match(match_id)
        return states[local_idx], labels[local_idx], masks[local_idx]

    @property
    def match_ids(self) -> List[int]:
        return list(self._match_ids)

    @property
    def match_counts(self) -> List[Tuple[int, int]]:
        return list(self._counts)


class SequenceDataset(Dataset):
    """Wrap StatsBombDataset into fixed-length sequences for temporal training.

    Each sample is a dict:
        frames:             [T, N, F]   state snapshots (T=seq_len, N=23, F=10)
        lengths:            scalar int  actual number of valid timesteps
        pass_receiver_xy:   [T, 2]      target pass end (x,y), team-relative
        pass_receiver_slot: [T]         int receiver slot, -1 if no pass in horizon
        shot_xg:            [T, 2]      [shot_flag, xg_value]
        turnover:           [T]         binary turnover flag
    """

    def __init__(
        self,
        data_root: str = "/home/jack/workspace/open-data/data",
        seq_len: int = 50,
        stride: int = 25,
        horizon_seconds: float = 5.0,
        max_matches: Optional[int] = None,
    ):
        super().__init__()
        self.base = StatsBombDataset(
            data_root=data_root,
            horizon_seconds=horizon_seconds,
            max_matches=max_matches,
        )
        self.seq_len = seq_len
        self.stride = stride

        # Build index of (match_id, start_idx, actual_len) for all sequences.
        self._index: List[Tuple[int, int, int]] = []
        for match_id, n_events in self.base.match_counts:
            if n_events == 0:
                continue
            starts = list(range(0, n_events, stride))
            for start in starts:
                actual_len = min(seq_len, n_events - start)
                if actual_len < 2:
                    continue
                self._index.append((match_id, start, actual_len))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        match_id, start, actual_len = self._index[idx]
        # Find global offset for this match.
        match_idx = self.base._match_ids.index(match_id)
        global_offset = self.base._cum_lengths[match_idx] - self.base._counts[match_idx][1]

        states, labels, masks = [], [], []
        for local in range(start, start + actual_len):
            s, l, m = self.base[global_offset + local]
            states.append(s)
            labels.append(l)
            masks.append(m)

        # Pad to seq_len.
        pad_len = self.seq_len - actual_len
        for _ in range(pad_len):
            states.append(torch.zeros_like(states[0]))
            labels.append({
                "pass_receiver": torch.zeros(3, dtype=torch.float32),
                "shot_score": torch.tensor(0.0, dtype=torch.float32),
                "turnover": torch.tensor(0.0, dtype=torch.float32),
            })
            masks.append(torch.zeros_like(masks[0]))

        frames = torch.stack(states, dim=0)            # [T, N, F]
        mask = torch.stack(masks, dim=0)              # [T, 22]

        pass_xy = torch.zeros(self.seq_len, 2, dtype=torch.float32)
        pass_slot = torch.full((self.seq_len,), -1, dtype=torch.long)
        shot_xg = torch.zeros(self.seq_len, 2, dtype=torch.float32)
        turnover = torch.zeros(self.seq_len, dtype=torch.float32)

        for t, lbl in enumerate(labels[:actual_len]):
            pr = lbl["pass_receiver"]
            pass_xy[t] = pr[:2]
            slot = int(pr[2].item()) if pr[2].item() >= 0 else -1
            # Guard against transient on-pitch counts > 11 per team.
            if slot >= 22:
                slot = -1
            pass_slot[t] = slot
            shot_xg[t, 0] = 1.0 if lbl["shot_score"].item() > 0 else 0.0
            shot_xg[t, 1] = lbl["shot_score"].item()
            turnover[t] = lbl["turnover"].item()

        return {
            "frames": frames,
            "lengths": torch.tensor(actual_len, dtype=torch.long),
            "mask": mask,
            "pass_receiver_xy": pass_xy,
            "pass_receiver_slot": pass_slot,
            "shot_xg": shot_xg,
            "turnover": turnover,
        }
