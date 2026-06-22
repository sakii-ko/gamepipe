from __future__ import annotations

import os

import numpy as np

from ..core import Clip, FrameSource, Manifest, cached_keyframes, survivors


def _survivors(filter_out):
    clips_m = Manifest(os.path.join(filter_out, "clips.jsonl"))
    stages = [Manifest(os.path.join(filter_out, f"{s}.jsonl"))
              for s in ("ocr", "hud") if os.path.exists(os.path.join(filter_out, f"{s}.jsonl"))]
    return [Clip.from_dict(r) for r in survivors(clips_m, *stages)]


def caption_stage(filter_out, ann_out, captioner):
    """Caption each survivor -> <id>/caption.txt. Own process (vLLM). Reuses the filter's
    keyframe cache (decode-once) — the caption model only sub-samples keyframes anyway."""
    kf_dir = os.path.join(filter_out, "kf")
    for clip in _survivors(filter_out):
        d = os.path.join(ann_out, clip.clip_id)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "caption.txt"), "w") as f:
            f.write(captioner.caption(cached_keyframes(kf_dir, clip)))


def depth_stage(filter_out, ann_out, depth_cam):
    """Depth/camera for each survivor -> <id>/geometry.npz. Own process (vggt)."""
    for clip in _survivors(filter_out):
        d = os.path.join(ann_out, clip.clip_id)
        os.makedirs(d, exist_ok=True)
        np.savez(os.path.join(d, "geometry.npz"), **depth_cam.estimate(FrameSource(clip).frames()))


def annotate_clips(clips, frames_of, captioner, depth_cam, out_dir):
    """One-process caption+depth (tests / mock backends)."""
    for clip in clips:
        d = os.path.join(out_dir, clip.clip_id)
        os.makedirs(d, exist_ok=True)
        frames = frames_of(clip)
        with open(os.path.join(d, "caption.txt"), "w") as f:
            f.write(captioner.caption(frames))
        np.savez(os.path.join(d, "geometry.npz"), **depth_cam.estimate(frames))
