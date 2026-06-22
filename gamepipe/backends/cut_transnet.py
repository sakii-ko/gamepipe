from __future__ import annotations

import numpy as np

from ..core.interfaces import CutDetector


class TransNetV2Detector(CutDetector):
    """soCzech/TransNetV2 scene-cut detection (48x27 input, 100-frame windows / stride 50 /
    central 50). Wraps the validated wforge detector. Streams [start, end) and downsamples on
    the fly (the 48x27 frames are tiny — bounded memory even for long runs)."""

    def __init__(self, threshold=0.5, weights=None, src=None, pkg=None):
        import sys

        from .. import config
        for p in (src or config.WFORGE_SRC, pkg or config.TRANSNET_PKG):
            if p not in sys.path:
                sys.path.insert(0, p)
        from wforge.segment.transnet import get_detector

        self._det = get_detector("transnetv2", weights=weights)
        self._det.setup()
        self.threshold = threshold

    def cuts_in_range(self, video, start, end):
        import av
        from PIL import Image

        small = []
        with av.open(video) as c:
            s = c.streams.video[0]
            fps, tb, st = float(s.average_rate), s.time_base, s.start_time or 0
            c.seek(st + int(start / fps / tb), stream=s, backward=True)
            for f in c.decode(s):
                if f.pts is None:
                    continue
                i = round(float((f.pts - st) * tb) * fps)
                if i < start:
                    continue
                if i >= end:
                    break
                small.append(np.asarray(Image.fromarray(f.to_ndarray(format="rgb24")).resize((48, 27))))
        if len(small) < 2:
            return []
        probs = self._det._predict(np.stack(small))
        return [int(i) for i in np.flatnonzero(probs > self.threshold)]
