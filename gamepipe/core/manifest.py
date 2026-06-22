from __future__ import annotations

import json
import os
from collections.abc import Iterator


class Manifest:
    """Append-only jsonl of clip rows. One file per stage; join with `survivors`."""

    def __init__(self, path: str):
        self.path = path

    def append(self, row: dict) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def __iter__(self) -> Iterator[dict]:
        if not os.path.exists(self.path):
            return
        with open(self.path) as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)

    def ids(self) -> set[str]:
        return {r["clip_id"] for r in self}


def survivors(clips: Manifest, *stages: Manifest) -> Iterator[dict]:
    """Clip rows kept by every stage that judged them (absent verdict = kept)."""
    keep: dict[str, bool] = {}
    for st in stages:
        for r in st:
            keep[r["clip_id"]] = keep.get(r["clip_id"], True) and bool(r.get("keep", True))
    for r in clips:
        if keep.get(r["clip_id"], True):
            yield r
