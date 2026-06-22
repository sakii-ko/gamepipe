# gamepipe

A data pipeline that turns raw game-playthrough recordings into clean, action-conditioned
training clips for world models. Two halves:

- **filter** — raw video + input logs → kept clip **index ranges** + per-clip action.
- **annotation** — kept clips → caption + depth/camera.

Nothing is re-encoded: the box has no NVENC, and decode (NVDEC/CPU) is cheap while CPU encode
is not — so a clip is only ever an index range into its source video, decoded on demand. The
jsonl **manifest** is the single source of truth.

## Layout

    gamepipe/
      core/          Clip, Manifest, FrameSource, backend ABCs   (the shared contracts)
      filter/        action -> segment -> ocr -> hud
      annotation/    caption, depth/camera
      backends/      pluggable heavy-model impls (+ mocks), via get_backend(kind, name)
    tests/           mock-based unit tests + GPU integration tests   (not committed)

## filter

A cheap→expensive funnel; each stage only sees the previous stage's survivors.

1. **action** (`filter/action.py`) — reconstruct per-frame input state from the raw event
   logs (`gamepad_axis/button/hat`, `key`, `mouse_*`) aligned to frames via `timeline.txt`
   (`obs_recording_started` = frame 0). Carry-forward absolute stick positions / held buttons
   / mouse-move. `frame_flags` → `move` / `cam` / `bad` per frame.
2. **segment** (`filter/segment.py`) — keep only pure locomotion+camera: a *bad* frame (any
   button / trigger / non-WASD key) poisons `[t-PRE, t+POST]` (default 0.5 s / 3 s). Remaining
   good runs are split at TransNetV2 scene cuts and windowed into clips of
   `MIN_FRAMES..MAX_FRAMES` model frames (default 81..401) at `decim = src_fps // model_fps`.
   Each clip also stores its **action** (`<clip_id>.action.json`).
3. **ocr** (`filter/ocr_filter.py`) — drop burned-in text overlays. `persistent_text` = max
   fraction of keyframes with a text box in the same grid cell (CRAFT detection only — we
   locate fixed-position text, never read it; in-world text moves and scores low).
4. **hud** (`filter/hud_filter.py`) — drop clips with game UI (health bar / minimap / …) via a
   small VLM. The VLM is the right tool here: an FPS weapon / cockpit is static-but-valid, so
   the cheap temporal-static pixel trick can't tell it from a HUD, but a VLM can.

## annotation

Survivor clips → per-clip `caption.txt` + `geometry.npz` (extrinsic / intrinsic / depth).
Frames come from `FrameSource` (seek/stream decode). Models are pluggable backends:
caption defaults to `qwen3.6-35b-a3b` (alt `qwen3-vl-8b`); depth/camera to `vggt-omega`.

## running

Each heavy model runs in its **own process** — EasyOCR's CUDA context and vLLM's spawned
EngineCore can't share one. The two orchestrators run the stages as subprocesses with a GPU
handoff between them; `-m gamepipe <stage>` runs a single stage in-process.

    gamepipe filter   <session> <video> --out DIR   # segment(+transnet) -> ocr -> hud
    gamepipe annotate --out DIR                      # depth -> caption
    gamepipe segment|ocr|hud|depth|caption --out DIR [...]   # one stage

Backends are chosen per stage (`--cuts/--ocr/--hud/--depth/--caption`); defaults are the real
ones. **Decode-once:** the filter caches each clip's keyframes (`DIR/kf/<clip>.npy`) on first
use, and ocr / hud / caption all read that cache instead of re-decoding (depth needs full
frames, decoded once on its own). The cache is transient — delete `DIR/kf` after annotation.

## backends

Every heavy model sits behind an ABC (`core/interfaces.py`) with a prod impl and a mock:

    get_backend("ocr", "easyocr")            get_backend("ocr", "mock")
    get_backend("hud", "qwen_vl")            get_backend("hud", "mock")
    get_backend("caption", "qwen")           get_backend("caption", "mock")
    get_backend("depth_camera", "vggt")      get_backend("depth_camera", "mock")
    get_backend("cut", "transnetv2")         get_backend("cut", "mock")

External paths + model ids resolve through `config.py`, overridable by env
(`GAMEPIPE_WFORGE_SRC`, `GAMEPIPE_TRANSNET_PKG`, `GAMEPIPE_HF`, `GAMEPIPE_CAPTION_MODEL`,
`GAMEPIPE_HUD_MODEL`).

## testing

    python -m pytest        # mock-based logic tests (no GPU)

GPU integration tests (real ocr/hud/caption/depth) live alongside and run when a GPU is
present. `tests/` is gitignored.

See `SPEC.md` for the full contract.
