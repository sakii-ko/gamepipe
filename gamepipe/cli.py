"""gamepipe CLI. Each heavy model runs in its OWN process (EasyOCR's CUDA context and vLLM's
spawned EngineCore can't share a process); `filter` / `annotate` orchestrate the stages as
subprocesses with a GPU handoff between them.

  gamepipe filter   <session> <video> --out DIR     # segment(+cuts) -> ocr -> hud
  gamepipe annotate --out DIR                        # depth -> caption
  gamepipe segment|ocr|hud|depth|caption ...         # a single stage, in this process
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time


def _gpu_free_mb():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"])
        return int(out.decode().splitlines()[0])
    except Exception:
        return 1 << 30                            # no nvidia-smi -> treat as unbounded


def _wait_gpu(min_mb=60000, tries=36, every=5):
    for _ in range(tries):
        if _gpu_free_mb() >= min_mb:
            return
        try:
            import psutil
            for p in psutil.process_iter(["name", "cmdline"]):
                nm = (p.info["name"] or "") + " ".join(p.info["cmdline"] or [])
                if ("EngineCore" in nm or "VLLM" in nm) and p.pid != os.getpid():
                    p.kill()
        except Exception:
            pass
        time.sleep(every)


def _sub(stage, *cli):
    subprocess.run([sys.executable, "-m", "gamepipe", stage, *cli], check=True)


def _ann_dir(a):
    return a.ann or os.path.join(a.out, "annotations")


def cmd_segment(a):
    from .backends import get_backend
    from .filter.run import segment_stage
    from .filter.segment import cut_fn_from
    cut_fn = None if a.cuts == "none" else cut_fn_from(get_backend("cut", a.cuts))
    clips = segment_stage(a.session, a.video, a.out, session_id=a.session_id, fps=a.fps,
                          decim=a.decim, cut_fn=cut_fn, max_frames=a.max_frames,
                          min_f=a.min_frames, max_f=a.clip_frames)
    print(f"segment: {len(clips)} clips -> {a.out}/clips.jsonl (+ actions/)")


def cmd_ocr(a):
    from .backends import get_backend
    from .filter.run import ocr_stage
    ocr_stage(a.out, get_backend("ocr", a.backend), thr=a.thr)
    print(f"ocr -> {a.out}/ocr.jsonl")


def cmd_hud(a):
    from .backends import get_backend
    from .filter.run import hud_stage
    hud_stage(a.out, get_backend("hud", a.backend))
    print(f"hud -> {a.out}/hud.jsonl")


def cmd_depth(a):
    from .annotation import depth_stage
    from .backends import get_backend
    depth_stage(a.out, _ann_dir(a), get_backend("depth_camera", a.backend, fps=a.fps))
    print(f"depth -> {_ann_dir(a)}/<id>/geometry.npz")


def cmd_caption(a):
    from .annotation import caption_stage
    from .backends import get_backend
    caption_stage(a.out, _ann_dir(a), get_backend("caption", a.backend))
    print(f"caption -> {_ann_dir(a)}/<id>/caption.txt")


def cmd_filter(a):
    seg = ["segment", a.session, a.video, "--out", a.out, "--fps", str(a.fps),
           "--decim", str(a.decim), "--cuts", a.cuts, "--min-frames", str(a.min_frames),
           "--clip-frames", str(a.clip_frames)]
    if a.session_id:
        seg += ["--session-id", a.session_id]
    if a.max_frames:
        seg += ["--max-frames", str(a.max_frames)]
    _sub(*seg)
    _wait_gpu(); _sub("ocr", "--out", a.out, "--backend", a.ocr)
    _wait_gpu(); _sub("hud", "--out", a.out, "--backend", a.hud)
    print(f"filter done -> survivors in {a.out} (clips ∧ ocr ∧ hud)")


def cmd_annotate(a):
    _wait_gpu(); _sub("depth", "--out", a.out, "--ann", _ann_dir(a), "--backend", a.depth, "--fps", str(a.fps))
    _wait_gpu(); _sub("caption", "--out", a.out, "--ann", _ann_dir(a), "--backend", a.caption)
    print(f"annotate done -> {_ann_dir(a)}")


def _add_seg_args(p):
    p.add_argument("--session-id", default=None)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--decim", type=int, default=2)
    p.add_argument("--max-frames", type=int, default=None, help="cap source frames (debug)")
    p.add_argument("--min-frames", type=int, default=81)
    p.add_argument("--clip-frames", type=int, default=401, help="max model frames per clip")
    p.add_argument("--cuts", default="transnetv2", choices=["transnetv2", "mock", "none"])


def build_parser():
    p = argparse.ArgumentParser(prog="gamepipe")
    sub = p.add_subparsers(dest="cmd", required=True)

    seg = sub.add_parser("segment"); seg.add_argument("session"); seg.add_argument("video")
    seg.add_argument("--out", required=True); _add_seg_args(seg); seg.set_defaults(fn=cmd_segment)

    ocr = sub.add_parser("ocr"); ocr.add_argument("--out", required=True)
    ocr.add_argument("--backend", default="easyocr"); ocr.add_argument("--thr", type=float, default=0.5)
    ocr.set_defaults(fn=cmd_ocr)

    hud = sub.add_parser("hud"); hud.add_argument("--out", required=True)
    hud.add_argument("--backend", default="qwen_vl"); hud.set_defaults(fn=cmd_hud)

    dep = sub.add_parser("depth"); dep.add_argument("--out", required=True)
    dep.add_argument("--ann", default=None); dep.add_argument("--backend", default="vggt")
    dep.add_argument("--fps", type=float, default=15.0); dep.set_defaults(fn=cmd_depth)

    cap = sub.add_parser("caption"); cap.add_argument("--out", required=True)
    cap.add_argument("--ann", default=None); cap.add_argument("--backend", default="qwen")
    cap.set_defaults(fn=cmd_caption)

    flt = sub.add_parser("filter"); flt.add_argument("session"); flt.add_argument("video")
    flt.add_argument("--out", required=True); _add_seg_args(flt)
    flt.add_argument("--ocr", default="easyocr"); flt.add_argument("--hud", default="qwen_vl")
    flt.set_defaults(fn=cmd_filter)

    ann = sub.add_parser("annotate"); ann.add_argument("--out", required=True)
    ann.add_argument("--ann", default=None); ann.add_argument("--fps", type=float, default=15.0)
    ann.add_argument("--depth", default="vggt"); ann.add_argument("--caption", default="qwen")
    ann.set_defaults(fn=cmd_annotate)
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    a.fn(a)
