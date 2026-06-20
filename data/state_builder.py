"""Helpers for building football state tensors from StatsBomb event data.

All coordinates are normalized to [-1, 1] on a 120 x 80 metre pitch. Both
regular halves are oriented so that the team that kicked off the first half
always attacks toward +x. Extra-time periods keep the same odd/even flip rule.
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

PITCH_LENGTH = 120.0
PITCH_WIDTH = 80.0

# Indices aligned with models.spatial_encoder feature layout.
IDX_X, IDX_Y, IDX_VX, IDX_VY = 0, 1, 2, 3
IDX_TEAM0, IDX_TEAM1 = 4, 5
IDX_POSITION = 6
IDX_POSS = 7
IDX_BALL = 8
IDX_ON_PITCH = 9

TURNOVER_TYPES = {"Interception", "Ball Recovery", "Dispossessed"}
BALL_CONTROL_TYPES = {
    "Pass",
    "Carry",
    "Ball Receipt*",
    "Shot",
    "Dribble",
    "Ball Recovery",
    "Interception",
    "Goal Keeper",
    "Clearance",
    "Miscontrol",
}


def _parse_timestamp(ts: str) -> float:
    """StatsBomb timestamp 'HH:MM:SS.mmm' -> seconds within the period."""
    h, m, s = ts.split(":")
    return int(h) * 3600.0 + int(m) * 60.0 + float(s)


def _clock_to_seconds(clock: str) -> float:
    """Lineup clock 'MM:SS' or 'MM:SS.mmm' -> total seconds."""
    m, s = clock.split(":")
    return int(m) * 60.0 + float(s)


def _period_base_seconds(period: int) -> float:
    """Seconds added before the start of each period.

    Period 1: 0, period 2: 45', period 3: 90', period 4: 105'.
    """
    if period <= 1:
        return 0.0
    if period == 2:
        return 45.0 * 60.0
    if period == 3:
        return 90.0 * 60.0
    if period == 4:
        return 105.0 * 60.0
    # Fall back for unexpected extra periods.
    return (90.0 + (period - 3) * 15.0) * 60.0


def _period_duration_seconds(period: int) -> float:
    """Nominal duration of a period in seconds."""
    return 45.0 * 60.0 if period <= 2 else 15.0 * 60.0


def _seconds_in_period(period: int, clock_seconds: float) -> float:
    return clock_seconds - _period_base_seconds(period)


def _orient_coords(
    x: float,
    y: float,
    period: int,
    length: float = PITCH_LENGTH,
    width: float = PITCH_WIDTH,
) -> Tuple[float, float]:
    """Rotate even periods by 180 degrees so the reference team always attacks +x."""
    if period % 2 == 0:
        x = length - x
        y = width - y
    return x, y


def _normalize(
    x: float, y: float, length: float = PITCH_LENGTH, width: float = PITCH_WIDTH
) -> Tuple[float, float]:
    """Map pitch coordinates to [-1, 1]."""
    return (2.0 * x / length) - 1.0, (2.0 * y / width) - 1.0


def _orient_and_normalize(
    x: float, y: float, period: int
) -> Tuple[float, float]:
    x, y = _orient_coords(x, y, period)
    return _normalize(x, y)


def _load_json(path: Path) -> list:
    return json.loads(path.read_text(encoding="utf-8"))


def load_events(match_id, data_root: str = "/home/jack/workspace/open-data/data") -> List[dict]:
    return _load_json(Path(data_root) / "events" / f"{match_id}.json")


def load_lineups(match_id, data_root: str = "/home/jack/workspace/open-data/data") -> List[dict]:
    return _load_json(Path(data_root) / "lineups" / f"{match_id}.json")


def _infer_teams(events: List[dict], lineups: List[dict]) -> Tuple[int, int]:
    """Return (team0_id, team1_id) where team0 is the first-half kickoff team."""
    team_ids = {entry["team_id"] for entry in lineups}
    for ev in events:
        if ev["type"]["name"] == "Pass":
            pass_info = ev.get("pass", {})
            if pass_info.get("type", {}).get("name") == "Kick Off" and ev["period"] == 1:
                team0 = ev["team"]["id"]
                team1 = (team_ids - {team0}).pop()
                return team0, team1
    # Fallback: use the possession team of the first event.
    for ev in events:
        if "possession_team" in ev:
            team0 = ev["possession_team"]["id"]
            team1 = (team_ids - {team0}).pop()
            return team0, team1
    # Last resort: first two teams in the lineups file.
    return tuple(sorted(team_ids))[:2]


def _parse_lineups(
    lineups: List[dict], max_period: int
) -> Dict[int, Dict]:
    """Parse lineups into per-player metadata.

    Returns a dict keyed by player_id with:
        team_id, position_timeline, on_pitch_timeline
    where timelines are lists of (period, start_seconds, end_seconds, [position_id]).
    """
    info: Dict[int, Dict] = {}
    for team in lineups:
        team_id = team["team_id"]
        for player in team["lineup"]:
            pid = player["player_id"]
            pos_timeline: List[Tuple[int, float, float, int]] = []
            on_timeline: List[Tuple[int, float, float]] = []
            for pos in player.get("positions", []):
                from_period = pos["from_period"]
                to_str = pos.get("to")
                # If no end time, the player stays on until the final whistle.
                to_period = (
                    pos.get("to_period", from_period)
                    if to_str is not None
                    else max_period
                )
                from_clock = _clock_to_seconds(pos["from"])
                to_clock = _clock_to_seconds(to_str) if to_str is not None else None
                position_id = pos.get("position_id", 0) or 0

                for p in range(from_period, to_period + 1):
                    start = (
                        _seconds_in_period(from_period, from_clock)
                        if p == from_period
                        else 0.0
                    )
                    if p == to_period:
                        if to_clock is None:
                            end = math.inf
                        else:
                            end = _seconds_in_period(to_period, to_clock)
                    else:
                        end = _period_duration_seconds(p)
                    pos_timeline.append((p, start, end, position_id))
                    on_timeline.append((p, start, end))
            info[pid] = {
                "team_id": team_id,
                "position_timeline": pos_timeline,
                "on_pitch_timeline": on_timeline,
            }
    return info


def _is_on_pitch(
    player_info: Dict, period: int, sec_in_period: float
) -> bool:
    for per, start, end in player_info["on_pitch_timeline"]:
        if per == period and start <= sec_in_period <= end:
            return True
    return False


def _current_position_id(
    player_info: Dict, period: int, sec_in_period: float
) -> int:
    for per, start, end, pos_id in player_info["position_timeline"]:
        if per == period and start <= sec_in_period <= end:
            return pos_id
    return 0


class MatchState:
    """Stateful parser that walks through one match's events and emits tensors.

    The returned state has shape (23, 10):
        row 0     : ball -> [x, y, 0, 0, 0, 0, 0, 0, 1, 1]
        rows 1-22 : players -> [x, y, vx, vy, team0, team1, position_id,
                                is_possession, ball, on_pitch]
    Rows are ordered by team: team0 slots 0-10, team1 slots 11-21.

    Labels are a dict:
        pass_receiver: (3,) -> [team_rel_x, team_rel_y, receiver_slot]
            receiver_slot is -1 if no pass in the horizon.
        shot_score: scalar -> statsbomb xG if a shot occurs within k seconds.
        turnover: scalar -> 1 if possession changes via interception/ball
            recovery/dispossessed within k seconds.

    Mask has shape (22,) and is 1 for players currently on the pitch.
    """

    def __init__(
        self,
        events: List[dict],
        lineups: List[dict],
        horizon_seconds: float = 5.0,
    ):
        self.events = events
        self.horizon = horizon_seconds
        self.team0_id, self.team1_id = _infer_teams(events, lineups)
        self.max_period = max((ev["period"] for ev in events), default=2)
        self.player_info = _parse_lineups(lineups, self.max_period)

        # Per-player normalized state.
        self._positions: Dict[int, np.ndarray] = {}
        self._velocities: Dict[int, np.ndarray] = {}

        self._ball_pos = np.zeros(2, dtype=np.float32)
        self._ball_controller: Optional[int] = None
        self._last_time: Optional[float] = None
        self._current_period = 1

        self._states: List[torch.Tensor] = []
        self._masks: List[torch.Tensor] = []
        self._labels: List[Dict[str, torch.Tensor]] = []

        self._build()

    def __len__(self) -> int:
        return len(self._states)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        return self._states[idx], self._labels[idx], self._masks[idx]

    def _team_sign(self, team_id: int) -> int:
        """+1 for team0 (reference), -1 for the opponent."""
        return 1 if team_id == self.team0_id else -1

    def _update_time(self, ev: dict) -> float:
        period = ev["period"]
        sec_in_period = _parse_timestamp(ev["timestamp"])
        t = _period_base_seconds(period) + sec_in_period
        self._current_period = period
        return t

    def _event_location(self, ev: dict) -> Optional[np.ndarray]:
        loc = ev.get("location")
        if loc is None or len(loc) < 2:
            return None
        x, y = _orient_and_normalize(float(loc[0]), float(loc[1]), ev["period"])
        return np.array([x, y], dtype=np.float32)

    def _carry_end_location(self, ev: dict) -> Optional[np.ndarray]:
        carry = ev.get("carry")
        if carry is None:
            return None
        end = carry.get("end_location")
        if end is None or len(end) < 2:
            return None
        x, y = _orient_and_normalize(float(end[0]), float(end[1]), ev["period"])
        return np.array([x, y], dtype=np.float32)

    def _pass_end_location(self, ev: dict) -> Optional[np.ndarray]:
        p = ev.get("pass", {})
        end = p.get("end_location")
        if end is None or len(end) < 2:
            return None
        x, y = _orient_and_normalize(float(end[0]), float(end[1]), ev["period"])
        return np.array([x, y], dtype=np.float32)

    def _propagate_players(self, dt: float) -> None:
        for pid in list(self._positions.keys()):
            self._positions[pid] = self._positions[pid] + self._velocities.get(
                pid, np.zeros(2)
            ) * dt

    def _ensure_on_pitch_initialized(self, period: int, sec_in_period: float) -> None:
        for pid, info in self.player_info.items():
            if _is_on_pitch(info, period, sec_in_period) and pid not in self._positions:
                self._positions[pid] = np.zeros(2, dtype=np.float32)
                self._velocities[pid] = np.zeros(2, dtype=np.float32)

    def _on_pitch_players(
        self, period: int, sec_in_period: float
    ) -> Tuple[List[int], List[int]]:
        """Return sorted on-pitch player ids for (team0, team1)."""
        team0, team1 = [], []
        for pid, info in self.player_info.items():
            if _is_on_pitch(info, period, sec_in_period):
                if info["team_id"] == self.team0_id:
                    team0.append(pid)
                else:
                    team1.append(pid)
        return sorted(team0), sorted(team1)

    def _player_slot(
        self, player_id: int, period: int, sec_in_period: float
    ) -> Optional[int]:
        """Return the state-row slot for a player at a given moment, if on pitch."""
        info = self.player_info.get(player_id)
        if info is None or not _is_on_pitch(info, period, sec_in_period):
            return None
        team0, team1 = self._on_pitch_players(period, sec_in_period)
        if info["team_id"] == self.team0_id:
            try:
                return team0.index(player_id)
            except ValueError:
                return None
        try:
            return 11 + team1.index(player_id)
        except ValueError:
            return None

    def _update_player_position(self, player_id: int, new_pos: np.ndarray, dt: float) -> None:
        old_pos = self._positions.get(player_id)
        if old_pos is None:
            self._velocities[player_id] = np.zeros(2, dtype=np.float32)
        elif dt > 1e-6:
            self._velocities[player_id] = (new_pos - old_pos) / dt
        else:
            # Keep existing velocity if time did not advance.
            if player_id not in self._velocities:
                self._velocities[player_id] = np.zeros(2, dtype=np.float32)
        self._positions[player_id] = new_pos

    def _step(self, ev: dict) -> None:
        t = self._update_time(ev)
        dt = 0.0 if self._last_time is None else t - self._last_time
        self._last_time = t

        period = ev["period"]
        sec_in_period = _parse_timestamp(ev["timestamp"])

        # Ensure newly-on players have state and propagate known players forward.
        self._ensure_on_pitch_initialized(period, sec_in_period)
        self._propagate_players(dt)

        # Update ball location if available.
        loc = self._event_location(ev)
        if loc is not None:
            self._ball_pos = loc

        # Update controller. A player in a ball-control event is the ball carrier.
        ev_type = ev["type"]["name"]
        player_id = ev.get("player", {}).get("id")
        if player_id is not None and ev_type in BALL_CONTROL_TYPES:
            self._ball_controller = player_id

        # Update the acting player's location from the event.
        if player_id is not None and loc is not None:
            self._update_player_position(player_id, loc, dt)
            # If this is a carry, also set a velocity toward the carry end so that
            # the next event sees a better estimate of the player's location.
            end_loc = self._carry_end_location(ev)
            if end_loc is not None:
                duration = ev.get("duration", 0.0)
                if duration > 1e-6:
                    self._velocities[player_id] = (end_loc - loc) / duration

    def _build_state(self, ev: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        period = ev["period"]
        sec_in_period = _parse_timestamp(ev["timestamp"])

        team0_pids, team1_pids = self._on_pitch_players(period, sec_in_period)
        team0_pids = team0_pids[:11]
        team1_pids = team1_pids[:11]

        n_players = 22
        player_tensor = np.zeros((n_players, 10), dtype=np.float32)
        mask = np.zeros(n_players, dtype=np.float32)

        def fill(rows, pids, team_onehot, offset: int = 0):
            for slot, pid in enumerate(pids):
                pos = self._positions.get(pid, np.zeros(2, dtype=np.float32))
                vel = self._velocities.get(pid, np.zeros(2, dtype=np.float32))
                info = self.player_info[pid]
                pos_id = _current_position_id(info, period, sec_in_period)
                is_poss = 1 if pid == self._ball_controller else 0
                rows[offset + slot] = [
                    pos[0],
                    pos[1],
                    vel[0],
                    vel[1],
                    team_onehot[0],
                    team_onehot[1],
                    float(pos_id),
                    float(is_poss),
                    0.0,          # ball flag
                    1.0,          # on_pitch flag
                ]
                mask[offset + slot] = 1.0

        fill(player_tensor, team0_pids, [1.0, 0.0], offset=0)
        fill(player_tensor, team1_pids, [0.0, 1.0], offset=11)

        ball_row = np.zeros(10, dtype=np.float32)
        ball_row[:2] = self._ball_pos
        ball_row[IDX_BALL] = 1.0
        ball_row[IDX_ON_PITCH] = 1.0
        state = np.concatenate([ball_row[None, :], player_tensor], axis=0)

        return torch.from_numpy(state), torch.from_numpy(mask)

    def _build_labels(self, idx: int) -> Dict[str, torch.Tensor]:
        ev = self.events[idx]
        current_team = ev.get("possession_team", {}).get("id", self.team0_id)
        team_sign = self._team_sign(current_team)
        start_t = _period_base_seconds(ev["period"]) + _parse_timestamp(ev["timestamp"])

        pass_receiver = np.zeros(3, dtype=np.float32)
        pass_receiver[2] = -1.0
        pass_found = False
        shot_score = np.array(0.0, dtype=np.float32)
        turnover = np.array(0.0, dtype=np.float32)

        for j in range(idx + 1, len(self.events)):
            fut = self.events[j]
            fut_period = fut["period"]
            fut_sec = _parse_timestamp(fut["timestamp"])
            fut_t = _period_base_seconds(fut_period) + fut_sec
            if fut_t - start_t > self.horizon:
                break

            fut_type = fut["type"]["name"]
            fut_team = fut.get("team", {}).get("id")
            fut_poss = fut.get("possession_team", {}).get("id")

            # Pass receiver: first pass by the current team in the horizon.
            if not pass_found and fut_type == "Pass" and fut_poss == current_team:
                end = self._pass_end_location(fut)
                if end is not None:
                    pass_receiver[0] = end[0] * team_sign
                    pass_receiver[1] = end[1] * team_sign
                    recip = fut.get("pass", {}).get("recipient", {}).get("id")
                    if recip is not None:
                        slot = self._player_slot(recip, fut_period, fut_sec)
                        pass_receiver[2] = float(slot) if slot is not None else -1.0
                    else:
                        pass_receiver[2] = -1.0
                    pass_found = True

            # Shot score: first shot by current team in the horizon.
            if shot_score.item() == 0.0 and fut_type == "Shot" and fut_poss == current_team:
                xg = fut.get("shot", {}).get("statsbomb_xg", 0.0)
                shot_score = np.array(float(xg), dtype=np.float32)

            # Turnover: possession changes via interception/ball recovery/dispossessed.
            if turnover.item() == 0.0 and fut_type in TURNOVER_TYPES:
                poss_changed = fut_poss is not None and fut_poss != current_team
                dispossessed_by_us = fut_type == "Dispossessed" and fut_team == current_team
                if poss_changed or dispossessed_by_us:
                    turnover = np.array(1.0, dtype=np.float32)

            # Early exit if all labels are resolved.
            if pass_found and shot_score.item() > 0.0 and turnover.item() > 0.0:
                break

        return {
            "pass_receiver": torch.from_numpy(pass_receiver),
            "shot_score": torch.from_numpy(shot_score),
            "turnover": torch.from_numpy(turnover),
        }

    def _build(self) -> None:
        for i, ev in enumerate(self.events):
            self._step(ev)
            state, mask = self._build_state(ev)
            labels = self._build_labels(i)
            self._states.append(state)
            self._masks.append(mask)
            self._labels.append(labels)


# Keep backward-compatible aliases for the helper names expected by tests.
def parse_timestamp(ts: str) -> float:
    return _parse_timestamp(ts)


def clock_to_seconds(clock: str) -> float:
    return _clock_to_seconds(clock)


def period_base_seconds(period: int) -> float:
    return _period_base_seconds(period)


def period_duration_seconds(period: int) -> float:
    return _period_duration_seconds(period)


def seconds_in_period(period: int, clock_seconds: float) -> float:
    return _seconds_in_period(period, clock_seconds)


def orient_and_normalize(x: float, y: float, period: int) -> Tuple[float, float]:
    return _orient_and_normalize(x, y, period)
