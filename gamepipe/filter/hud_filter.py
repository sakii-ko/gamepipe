from __future__ import annotations


def keep(hud_vlm, frames) -> bool:
    return not hud_vlm.has_hud(frames)
