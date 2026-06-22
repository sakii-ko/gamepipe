from __future__ import annotations

from abc import ABC, abstractmethod


class DepthCameraBackend(ABC):
    @abstractmethod
    def estimate(self, frames) -> dict: ...     # {extrinsic, intrinsic, depth}


class Captioner(ABC):
    @abstractmethod
    def caption(self, frames) -> str: ...


class OcrDetector(ABC):
    @abstractmethod
    def detect(self, frame) -> list: ...        # text-box positions (no recognition)


class HudVLM(ABC):
    @abstractmethod
    def has_hud(self, frames) -> bool: ...


class CutDetector(ABC):
    @abstractmethod
    def cuts_in_range(self, video, start, end) -> list: ...   # cut frame indices relative to start

