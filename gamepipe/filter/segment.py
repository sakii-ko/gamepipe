from __future__ import annotations

import numpy as np

from ..core import Clip
from .action import ActionParams, ActionTrack, frame_flags, reconstruct


def poison_mask(bad: np.ndarray, pre: int, post: int) -> np.ndarray:
    n = len(bad)
    mask = np.zeros(n, bool)
    for i in np.flatnonzero(bad):
        mask[max(0, i - pre):min(n, i + post + 1)] = True
    return mask


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def good_runs(flags: dict, pre: int, post: int) -> list[tuple[int, int]]:
    good = (flags["move"] | flags["cam"]) & ~poison_mask(flags["bad"], pre, post)
    return _runs(good)


def split_cuts(run: tuple[int, int], cuts) -> list[tuple[int, int]]:
    s, e = run
    pts = [s, *(c for c in sorted(cuts) if s < c < e), e]
    return list(zip(pts, pts[1:]))


def window(start: int, end: int, decim: int, min_f: int, max_f: int) -> list[tuple[int, int]]:
    out, off, n = [], start, (end - start) // decim
    while n >= min_f:
        take = n if n <= max_f else max_f
        out.append((off, off + take * decim))
        off += take * decim
        n -= take
    return out


def cut_fn_from(detector):
    """Wrap a CutDetector into segment's cut_fn (source, run) -> source-frame cut indices."""
    def cut_fn(source, run):
        return [run[0] + c for c in detector.cuts_in_range(source, run[0], run[1])]
    return cut_fn


def segment_session(session_dir, source, fps, n_frames, decim, cut_fn, session_id="s",
                    params=ActionParams(), pre_s=0.5, post_s=3.0, min_f=81, max_f=401):
    track = reconstruct(session_dir, fps, n_frames)
    return clips_from_track(track, source, fps, decim, cut_fn, session_id,
                            params, pre_s, post_s, min_f, max_f)


def clips_from_track(track: ActionTrack, source, fps, decim, cut_fn, session_id="s",
                     params=ActionParams(), pre_s=0.5, post_s=3.0, min_f=81, max_f=401):
    flags = frame_flags(track, params)
    pre, post = round(pre_s * fps), round(post_s * fps)
    out = []
    for run in good_runs(flags, pre, post):
        for sub in split_cuts(run, cut_fn(source, run)):
            for s, e in window(*sub, decim, min_f, max_f):
                cid = f"{session_id}_{s}_{e}"
                out.append((Clip(cid, source, s, e, decim), track.clip_action(s, e, decim)))
    return out
