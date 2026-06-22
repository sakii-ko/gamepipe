# gamepipe — spec

Real game-footage data pipeline. Two halves:
- **filter**: raw video + input logs → kept clip **index ranges** (no re-encode — the box has no NVENC).
- **annotation**: kept clips → caption + depth/camera.

The **manifest** (jsonl, one row per clip) is the single source of truth. Filter never
materializes video; every video stage decodes on demand (NVDEC/CPU decode is cheap, only
encode is the missing-NVENC bottleneck).

## core (`core/`)

**Clip** — the unit, a view into a source video, never materialized.

    Clip(clip_id, source, start, end, decim)
      start, end : source frame range [start, end)
      decim      : source_fps // model_fps    (30 → 15 ⇒ 2)
      n_frames   = (end - start) // decim      # model frames
      src_frame(k) = start + decim * k
    bounds: MIN_FRAMES ≤ n_frames ≤ MAX_FRAMES   (params; default 81 / 401)

**Manifest** — jsonl; stages accrete fields onto a clip row:

    {clip_id, source, start, end, decim, n_frames,
     ocr?: {keep, persistent_text}, hud?: {keep},
     caption?, depth?, camera?}
  survivor ⇔ kept by every applied filter stage.

**ClipAction** — every kept clip stores its action (the conditioning signal). The session
ActionTrack sliced to [start, end) at `decim`, one entry per model frame. Written by segment
as a per-clip sidecar `<clip_id>.action.json`: `{axes[n,6], mouse[n,2], buttons[n], keys[n]}`
(axes = lstick xy / rstick xy / lt / rt; buttons/keys = held ids per frame). The manifest row
references it.

**FrameSource** — seek-decode of a Clip's model frames; bounded memory; no encode. Benchmark-
decided (see BENCHMARK.md): **PyAV** keyframe-seek + decode-forward (fastest CPU decoder, the
data is 4:4:4 H.264 so NVDEC is out); **decord/torchcodec `get_batch`** for sparse
`.keyframes(n)` (OCR/HUD/caption); `.frames()` for depth. Run **16–24 worker processes/node,
1 thread each** — throughput plateaus there (~1.3 clips/s/node) and the box is CPU/memory-
bound, not NAS-bound; scale with nodes, not workers. Never loop per-frame PyAV seeks.

**Backend ABCs** (`core/interfaces.py`) — every heavy model is pluggable; each has a prod impl
+ a mock. Factory `get_backend(kind, name)`.

    DepthCameraBackend.estimate(frames) -> {extrinsic, intrinsic, depth}
    Captioner.caption(frames)           -> str
    OcrDetector.detect(frame)           -> [box]          # CRAFT text-box POSITIONS only —
                                                          # no recognition / no conf filter;
                                                          # we locate persistent text, not read it
    HudVLM.has_hud(frames)              -> bool           # gameplay UI present?

## filter (`filter/`)

Pipeline **segment → ocr → hud**, cheap→expensive; each stage only sees survivors.

### 1. segment — raw session → candidate clips
Streaming over one session (long video + raw event logs); emits Clip index ranges.

- **action reconstruction** (`action.py`): raw logs (`gamepad_axis/button/hat`, `key`,
  `mouse_move/wheel`) + `timeline.txt` → per-frame state (carry-forward absolute stick
  position, held buttons, mouse-move). This is what iwm pre-baked as `prompts`; we build it
  for raw. Validated by comparing to iwm `prompts` on an overlapping session.
- **classify** per frame: good = stick locomotion/camera only (lstick OR rstick / WASD OR
  mouse-move), no button, no trigger, no click. bad = any of those, or long-idle.
- **poison**: a bad frame at t poisons [t − PRE, t + POST] (PRE=0.5s, POST=3s, both params).
  Remaining contiguous good runs = candidate spans.
- **cut split** (`transnet.py`): TransNetV2 (windowed, bounded memory) splits each good run
  at scene cuts.
- **window**: each cut-free good run → clips of n_frames ∈ [MIN_FRAMES, MAX_FRAMES] at decim.
  Emit a manifest row + write each clip's **action sidecar** (ActionTrack sliced to its frames).

params: PRE, POST, IDLE_SEC, MIN_FRAMES=81, MAX_FRAMES=401, MODEL_FPS=15, cut_thresh.

### 2. ocr (`ocr_filter.py`) — text watermark / text-HUD
OcrDetector over clip keyframes → `persistent_text` (max fraction of keyframes with a box in
the same grid cell). Drop if ≥ PERSIST_THR. Fixed-position text = overlay; in-world text moves
with the camera and scores low.

### 3. hud (`hud_filter.py`) — graphical HUD
HudVLM over clip keyframes → drop if gameplay UI present (health bar / minimap …). Small VLM;
distinguishes UI from an FPS weapon / cockpit, which the temporal-static pixel trick cannot.

### runner
Per session: reconstruct action from logs + stream-decode source once for TransNet → emit
clips → ocr → hud → write survivors. Resumable (skip clips already in the manifest).

## annotation (`annotation/`)

Survivor clips → per-clip outputs; frames via FrameSource (seek/stream, no materialization).

- **caption** (`caption.py`): Captioner, default `qwen3.6-35b-a3b`, alt `qwen3-vl-8b`.
- **depth_camera** (`depth_camera.py`): DepthCameraBackend, default `vggt-omega`. Pluggable
  for stronger backends later.

## cli

    gamepipe filter   <session-dir>  --out manifest.jsonl   # segment + ocr + hud
    gamepipe annotate <manifest>     --out <dir>            # caption + depth/camera
  all thresholds exposed as flags / config.

## testing — `tests/` (gitignored, never committed)
Mock backends for fast logic tests (poison radius, windowing, persistent_text, manifest);
real-model GPU integration tests where the model is the subject (ocr / hud / depth / caption).

## style
Concise (Karpathy). Let names + structure carry meaning; minimal comments, short docstrings
only where non-obvious. Detailed README.

## build order (each independently testable)
core → action reconstruction (validate vs iwm prompts) → segment(+transnet+window) →
ocr → hud-vlm → annotation.
