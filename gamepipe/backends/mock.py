from __future__ import annotations

import numpy as np

from ..core.interfaces import Captioner, CutDetector, DepthCameraBackend, HudVLM, OcrDetector


class MockOcr(OcrDetector):
    def __init__(self, boxes=None):
        self._boxes = boxes or []

    def detect(self, frame):
        return list(self._boxes)


class MockHud(HudVLM):
    def __init__(self, hud=False):
        self._hud = hud

    def has_hud(self, frames):
        return self._hud


class MockCutDetector(CutDetector):
    def __init__(self, cuts=None):
        self._cuts = cuts or []                  # cuts relative to the range start

    def cuts_in_range(self, video, start, end):
        return [c for c in self._cuts if 0 < c < end - start]


class MockCaptioner(Captioner):
    def caption(self, frames):
        return "a person in a rendered outdoor scene"


class MockDepthCamera(DepthCameraBackend):
    def estimate(self, frames):
        n = len(frames)
        return {"extrinsic": np.zeros((n, 4, 4), np.float32),
                "intrinsic": np.zeros((n, 3, 3), np.float32),
                "depth": np.zeros((n, 1, 1), np.float32)}
