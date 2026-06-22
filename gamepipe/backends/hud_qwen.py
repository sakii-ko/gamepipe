from __future__ import annotations

import json

import numpy as np

from ..core.interfaces import HudVLM

PROMPT = """You are a strict filter for a world-model training set of clean AAA gameplay. \
You see keyframes (temporal order) from ONE clip. Detect a GAME HUD / UI OVERLAY drawn on \
top of the world: health/mana/stamina/shield bars, a minimap or compass, an ability/spell/ \
weapon-selection wheel, a quest tracker or objective text, an ammo/score/money counter, a \
hotbar or item slots, big floating damage numbers, or an on-screen interaction/button prompt.

NOT a HUD (answer no): a first-person WEAPON, hands/arms, or a vehicle cockpit/steering \
wheel/dashboard — these are rendered 3D world. In-world signs or text on objects are not \
HUD. A tiny lone crosshair dot is not enough.

Answer present=yes only if real UI is overlaid on a noticeable part of the screen in at \
least some frames; when borderline, prefer yes. Output ONLY the JSON."""

SCHEMA = {"type": "object", "properties": {"hud": {"type": "object", "properties": {
    "reason": {"type": "string", "maxLength": 80},
    "present": {"type": "string", "enum": ["yes", "no"]}},
    "required": ["reason", "present"]}}, "required": ["hud"]}

from .. import config  # noqa: E402

DEFAULT_MODEL = config.HUD_MODEL


class QwenHud(HudVLM):
    def __init__(self, model=DEFAULT_MODEL, n_keyframes=8, max_pixels=401408,
                 gpu_memory_utilization=0.45):
        from vllm import LLM, SamplingParams
        from vllm.config import StructuredOutputsConfig
        from vllm.sampling_params import StructuredOutputsParams

        self.n_keyframes = n_keyframes
        self.llm = LLM(model=model, trust_remote_code=True,
                       limit_mm_per_prompt={"image": n_keyframes}, enforce_eager=True,
                       max_num_seqs=64, max_model_len=16384,
                       gpu_memory_utilization=gpu_memory_utilization,
                       mm_processor_kwargs={"max_pixels": max_pixels},
                       structured_outputs_config=StructuredOutputsConfig(
                           backend="xgrammar", disable_any_whitespace=True))
        self.sp = SamplingParams(temperature=0.0, max_tokens=96,
                                 structured_outputs=StructuredOutputsParams(json=SCHEMA))

    def _msgs(self, frames):
        from PIL import Image
        n = self.n_keyframes
        idx = np.linspace(0, len(frames) - 1, min(n, len(frames))).round().astype(int)
        imgs = [Image.fromarray(np.asarray(frames[i])[..., :3].astype("uint8")) for i in idx]
        content = [{"type": "image_pil", "image_pil": im} for im in imgs]
        content.append({"type": "text", "text": "Output ONLY the JSON."})
        return [{"role": "system", "content": PROMPT}, {"role": "user", "content": content}]

    def has_hud(self, frames) -> bool:
        out = self.llm.chat([self._msgs(frames)], sampling_params=self.sp, use_tqdm=False)
        try:
            return json.loads(out[0].outputs[0].text)["hud"]["present"] == "yes"
        except Exception:
            return False
