from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime

import numpy as np

N_AXES = 6
MOVE_AXES, CAM_AXES, TRIG_AXES = (0, 1), (2, 3), (4, 5)
WASD = {"w", "a", "s", "d"}
_MODS = {"ctrl", "alt", "shift"}
_TS = "%Y-%m-%d %H:%M:%S.%f"


def _split(line: str):
    return datetime.strptime(line[:23], _TS), line[25:].strip()


def _events(path: str):
    if os.path.exists(path):
        for ln in open(path):
            if ln.strip():
                yield _split(ln)


def recording_start(session_dir: str) -> datetime:
    lines = [ln for ln in open(f"{session_dir}/timeline.txt") if ln.strip()]
    for ln in lines:
        if "obs_recording_started" in ln:
            return _split(ln)[0]
    return _split(lines[0])[0]


@dataclass
class ActionParams:
    deadzone: float = 0.3       # stick deflection counted as active
    trig_on: float = -0.5       # trigger value above this (rest = -1) = pressed


@dataclass
class ActionTrack:
    axes: np.ndarray            # [n, 6]  carry-forward positions (sticks 0, triggers -1)
    buttons: list               # [n]  frozenset of held gamepad button ids
    mouse: np.ndarray           # [n, 2]  summed dx, dy per frame
    keys: list                  # [n]  frozenset of held key names

    def clip_action(self, start: int, end: int, decim: int) -> dict:
        idx = list(range(start, end, decim))
        return {
            "axes": np.round(self.axes[start:end:decim], 4).tolist(),
            "mouse": np.round(self.mouse[start:end:decim], 2).tolist(),
            "buttons": [sorted(self.buttons[i]) for i in idx],
            "keys": [sorted(self.keys[i]) for i in idx],
        }


def _carry(events, n, default):
    out = [default] * n
    cur, j, evs = default, 0, sorted(events)
    for f in range(n):
        while j < len(evs) and evs[j][0] <= f:
            cur = evs[j][1]
            j += 1
        out[f] = cur
    return out


def _hold(events, n):
    evs = sorted(events)
    held, out, j = set(), [], 0
    for f in range(n):
        while j < len(evs) and evs[j][0] <= f:
            _, item, down = evs[j]
            held.add(item) if down else held.discard(item)
            j += 1
        out.append(frozenset(held))
    return out


def reconstruct(session_dir: str, fps: float, n_frames: int) -> ActionTrack:
    start = recording_start(session_dir)
    frame = lambda ts: round((ts - start).total_seconds() * fps)  # noqa: E731

    axes = np.zeros((n_frames, N_AXES), np.float32)
    axes[:, 4:] = -1.0
    by_axis = [[] for _ in range(N_AXES)]
    for ts, p in _events(f"{session_dir}/gamepad_axis_0.txt"):
        name, _, val = p.split(",")
        by_axis[int(name.split("_")[1])].append((frame(ts), float(val)))
    for a in range(N_AXES):
        axes[:, a] = _carry(by_axis[a], n_frames, -1.0 if a in TRIG_AXES else 0.0)

    buttons = _hold(((frame(ts), p.split(",")[0], p.split(",")[1] == "d")
                     for ts, p in _events(f"{session_dir}/gamepad_button_0.txt")), n_frames)
    keys = _hold(((frame(ts), p.split(",")[0], p.split(",")[1] == "KEY_DOWN")
                  for ts, p in _events(f"{session_dir}/key_0.txt")), n_frames)

    mouse = np.zeros((n_frames, 2), np.float32)
    for ts, p in _events(f"{session_dir}/mouse_move_0.txt"):
        f = frame(ts)
        if 0 <= f < n_frames:
            dx, dy = p.split(",")[:2]
            mouse[f] += (float(dx), float(dy))

    return ActionTrack(axes, buttons, mouse, keys)


def frame_flags(track: ActionTrack, p: ActionParams = ActionParams()) -> dict:
    """Per-frame bool arrays: move / cam (wanted) and bad (button/trigger/non-WASD key)."""
    a = track.axes
    n = len(a)
    move = np.hypot(a[:, MOVE_AXES[0]], a[:, MOVE_AXES[1]]) > p.deadzone
    cam = np.hypot(a[:, CAM_AXES[0]], a[:, CAM_AXES[1]]) > p.deadzone
    trig = (a[:, TRIG_AXES[0]] > p.trig_on) | (a[:, TRIG_AXES[1]] > p.trig_on)
    btn = np.fromiter((bool(b) for b in track.buttons), bool, n)
    lower = [{k.lower() for k in ks} for ks in track.keys]
    kb_move = np.fromiter((bool(s & WASD) for s in lower), bool, n)
    kb_act = np.fromiter((bool(s - WASD - _MODS) for s in lower), bool, n)
    mouse_cam = np.hypot(track.mouse[:, 0], track.mouse[:, 1]) > 0
    return {"move": move | kb_move, "cam": cam | mouse_cam, "bad": btn | trig | kb_act}
