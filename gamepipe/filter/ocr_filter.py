from __future__ import annotations

from collections import Counter

GRID = 6


def _cell(box, h, w):
    xmin, xmax, ymin, ymax = box
    return int((ymin + ymax) / 2 / h * GRID), int((xmin + xmax) / 2 / w * GRID)


def persistent_text(detector, frames) -> float:
    """Max fraction of frames carrying a text box in the same grid cell = fixed-position
    burned-in text. In-world text scores high only when the camera is near-static — which is
    fine to drop anyway."""
    cells, k = Counter(), 0
    for f in frames:
        h, w = f.shape[:2]
        for c in {_cell(b, h, w) for b in detector.detect(f)}:
            cells[c] += 1
        k += 1
    return max(cells.values()) / k if cells and k else 0.0


def keep(detector, frames, thr: float = 0.5) -> bool:
    return persistent_text(detector, frames) < thr
