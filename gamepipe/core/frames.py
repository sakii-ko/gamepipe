from __future__ import annotations

import numpy as np


def _pyav_range(path, start, end, decim):
    """Seek to the keyframe at/before `start`, decode forward, keep [start, end) every decim-th.
    Frame index = round((pts - start_time) * time_base * fps) — 0-indexed, matching decord."""
    import av

    out = []
    with av.open(path) as c:
        s = c.streams.video[0]
        fps, tb, st = float(s.average_rate), s.time_base, s.start_time or 0
        c.seek(st + int(start / fps / tb), stream=s, backward=True)
        for f in c.decode(s):
            if f.pts is None:
                continue
            i = round(float((f.pts - st) * tb) * fps)
            if i < start:
                continue
            if i >= end:
                break
            if (i - start) % decim == 0:
                out.append(f.to_ndarray(format="rgb24"))
    return out


class FrameSource:
    """Decode-on-demand Clip frames. frames()=PyAV sequential; keyframes(n)=decord get_batch
    sparse (~3.7x cheaper, identical)."""

    def __init__(self, clip):
        self.clip = clip

    def frames(self):
        c = self.clip
        return _pyav_range(c.source, c.start, c.end, c.decim)

    def keyframes(self, n):
        import decord

        c = self.clip
        if c.n_frames <= n:
            return self.frames()
        mi = np.linspace(0, c.n_frames - 1, n).round().astype(int)
        return list(decord.VideoReader(c.source).get_batch([c.src_frame(int(m)) for m in mi]).asnumpy())


def cached_keyframes(cache_dir, clip, n=16):
    """Decode-once: the first stage to need a clip's keyframes decodes + caches them; ocr / hud
    / caption then reuse the cache instead of re-decoding the clip."""
    import os

    if cache_dir is None:
        return FrameSource(clip).keyframes(n)
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{clip.clip_id}.npy")
    if os.path.exists(path):
        return list(np.load(path))
    kf = np.stack(FrameSource(clip).keyframes(n))
    np.save(path, kf)
    return list(kf)
