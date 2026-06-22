from __future__ import annotations

import json
import os

from ..core import Clip, Manifest, cached_keyframes, survivors
from . import hud_filter, ocr_filter
from .action import reconstruct
from .segment import clips_from_track


def _video_len(path):
    import decord

    return len(decord.VideoReader(path))


def segment_stage(session_dir, video, out_dir, *, session_id=None, fps=30.0, decim=2,
                  cut_fn=None, max_frames=None, **seg_kw):
    """Raw session -> clips.jsonl + per-clip action sidecars. No GPU."""
    os.makedirs(os.path.join(out_dir, "actions"), exist_ok=True)
    session_id = session_id or os.path.basename(session_dir.rstrip("/"))
    n = _video_len(video)
    track = reconstruct(session_dir, fps, min(n, max_frames) if max_frames else n)
    clips = clips_from_track(track, video, fps, decim, cut_fn or (lambda src, run: []),
                             session_id=session_id, **seg_kw)
    man = Manifest(os.path.join(out_dir, "clips.jsonl"))
    for clip, action in clips:
        man.append(clip.to_dict())
        json.dump(action, open(os.path.join(out_dir, "actions", f"{clip.clip_id}.action.json"), "w"))
    return [c for c, _ in clips]


def _clips(out_dir):
    return [Clip.from_dict(r) for r in Manifest(os.path.join(out_dir, "clips.jsonl"))]


def _kf_dir(out_dir):
    return os.path.join(out_dir, "kf")


def ocr_stage(out_dir, ocr, *, thr=0.5):
    """Run the ocr filter over clips.jsonl -> ocr.jsonl. Own process (heavy model)."""
    m = Manifest(os.path.join(out_dir, "ocr.jsonl"))
    for clip in _clips(out_dir):
        pt = ocr_filter.persistent_text(ocr, cached_keyframes(_kf_dir(out_dir), clip))
        m.append({"clip_id": clip.clip_id, "keep": pt < thr, "persistent_text": round(pt, 3)})


def hud_stage(out_dir, hud):
    """Run the hud filter over the ocr survivors -> hud.jsonl. Own process (vLLM)."""
    stages = [Manifest(os.path.join(out_dir, "ocr.jsonl"))] if \
        os.path.exists(os.path.join(out_dir, "ocr.jsonl")) else []
    keep = {r["clip_id"] for r in survivors(Manifest(os.path.join(out_dir, "clips.jsonl")), *stages)}
    m = Manifest(os.path.join(out_dir, "hud.jsonl"))
    for clip in _clips(out_dir):
        if clip.clip_id in keep:
            m.append({"clip_id": clip.clip_id,
                      "keep": not hud.has_hud(cached_keyframes(_kf_dir(out_dir), clip))})


def run_filter(session_dir, video, out_dir, *, ocr=None, hud=None, ocr_thr=0.5, **seg_kw):
    """Convenience: all stages in one process (tests / mock or CPU backends only — a real vLLM
    backend must run in its own process; use the stages + the CLI then)."""
    clips = segment_stage(session_dir, video, out_dir, **seg_kw)
    if ocr is not None:
        ocr_stage(out_dir, ocr, thr=ocr_thr)
    if hud is not None:
        hud_stage(out_dir, hud)
    return clips
