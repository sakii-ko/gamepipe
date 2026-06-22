from __future__ import annotations

import numpy as np

from ..core.interfaces import OcrDetector


class EasyOcrDetector(OcrDetector):
    """CRAFT text detection only (no recognition) — returns box positions [xmin,xmax,ymin,ymax]."""

    def __init__(self, langs=("en",), text_threshold=0.4, low_text=0.3):
        import easyocr

        self.reader = easyocr.Reader(list(langs))
        self.text_threshold, self.low_text = text_threshold, low_text

    def detect(self, frame):
        boxes = self.reader.detect(np.asarray(frame), text_threshold=self.text_threshold,
                                   low_text=self.low_text)[0][0]
        return [[int(v) for v in b] for b in boxes]
