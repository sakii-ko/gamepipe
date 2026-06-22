from __future__ import annotations

from ..core.interfaces import DepthCameraBackend


class _Frames:
    def __init__(self, arrs):
        self._arrs = arrs

    def frames(self):
        return self._arrs


class _Clip:
    video_uri = ""

    def __init__(self, fps):
        self.fps = fps


class VggtBackend(DepthCameraBackend):
    """VGGT-Omega-Long direct path (loop_enable=False, single forward; clips are <=401 frames).
    Wraps the validated wforge reference backend; returns extrinsic [T,4,4] c2w / intrinsic [T,4]
    / metric depth [T,Hd,Wd]."""

    def __init__(self, fps=15.0, src=None, **overrides):
        import sys

        from .. import config
        src = src or config.WFORGE_SRC
        if src not in sys.path:
            sys.path.insert(0, src)
        from wforge.heavy.camera import ReferenceCameraBackend

        self._fps = fps
        self._be = ReferenceCameraBackend(which="vggt", **overrides)
        self._be.setup()

    def estimate(self, frames):
        out = self._be.estimate(_Clip(self._fps), _Frames(list(frames)))
        return {"extrinsic": out["cam_T_world"], "intrinsic": out["intrinsics"], "depth": out["depth"]}
