from __future__ import annotations

import importlib

_BACKENDS = {
    ("ocr", "easyocr"): "gamepipe.backends.ocr_easyocr:EasyOcrDetector",
    ("ocr", "mock"): "gamepipe.backends.mock:MockOcr",
    ("hud", "qwen_vl"): "gamepipe.backends.hud_qwen:QwenHud",
    ("hud", "mock"): "gamepipe.backends.mock:MockHud",
    ("caption", "qwen"): "gamepipe.backends.caption_qwen:QwenCaptioner",
    ("caption", "mock"): "gamepipe.backends.mock:MockCaptioner",
    ("depth_camera", "vggt"): "gamepipe.backends.depth_vggt:VggtBackend",
    ("depth_camera", "mock"): "gamepipe.backends.mock:MockDepthCamera",
    ("cut", "transnetv2"): "gamepipe.backends.cut_transnet:TransNetV2Detector",
    ("cut", "mock"): "gamepipe.backends.mock:MockCutDetector",
}


def get_backend(kind: str, name: str, **kw):
    mod, cls = _BACKENDS[(kind, name)].split(":")
    return getattr(importlib.import_module(mod), cls)(**kw)
