from __future__ import annotations

from dataclasses import asdict, dataclass

_FIELDS = ("clip_id", "source", "start", "end", "decim")


@dataclass(frozen=True)
class Clip:
    clip_id: str
    source: str
    start: int          # source frame, inclusive
    end: int            # source frame, exclusive
    decim: int          # source_fps // model_fps

    @property
    def n_frames(self) -> int:
        return (self.end - self.start) // self.decim

    def src_frame(self, k: int) -> int:
        return self.start + self.decim * k

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Clip:
        return cls(*(d[f] for f in _FIELDS))
