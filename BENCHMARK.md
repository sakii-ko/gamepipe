# Decode benchmark — index-range vs materialize

Settles FrameSource design. Run on the real raw recordings (1080p30, **H.264 4:4:4**,
**GOP 250**, ~15 Mbps, 18–32 min) over 30 random clips.

## Decision: keep clips as index ranges, decode-on-demand. Do NOT materialize short mp4s.

Materialize = the same decode **plus** a CPU x264 encode (~30 s/core, **no NVENC on the box**)
**plus** ~330 GB for 22k clips; stream-copy (`-c:v copy`) is cheap but keyframe-aligned only
(GOP 250 ⇒ ±8 s cut error). Materialization only pays off under many-epoch read-back.

## Findings

- **No GPU path.** 4:4:4 chroma ⇒ NVDEC `h264_cuvid` fails; torchcodec/decord builds are
  CPU-only; NVENC absent. CPU decode only.
- **Backend: PyAV** (keyframe-seek + decode-forward) — fastest true single-thread decoder,
  scales 4.8× to 8 threads. decord/torchcodec `get_batch` for **sparse keyframe sampling**
  only. **Never** loop PyAV per-frame seeks (re-decodes overlapping GOPs, slower than full).
- **GOP 250** ⇒ sampling 12 frames (~6–7 s) is barely cheaper than a full clip decode
  (~8–9 s); each random seek pays ~125 ramp frames (~23% extra).
- **Concurrency: 16–24 workers/node, 1 thread each.** Throughput plateaus ~1.2–1.5 clips/s/
  node and *declines* past 32. Warm-cache (no NAS) and cold give the same ceiling ⇒
  **CPU/memory-bound, not NAS-bound** (~35 MB/s NAS draw at W=32). Scale with nodes.
- **Seek vs stream:** break-even ≈ 49 clips/video; survivors/video ≪ 49 ⇒ per-clip seek wins.
- **~22k clips ≈ 4.5–5 h on one node**, near-linear across nodes.

(The earlier "NAS saturation" symptom was a different pattern — whole-clip decode + 80
workers + ffprobe storms — not this seek-decode access.)
