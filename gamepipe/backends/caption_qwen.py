from __future__ import annotations

import numpy as np

from ..core.interfaces import Captioner

PROMPT = """You are an expert video annotator and a master of visual descriptive writing. \
For every video provided in the user message, you write a single detailed, continuous, and \
evocative caption, and you always respond with the caption and nothing else.

Write a single-paragraph description that seamlessly integrates the following elements:
1. Subject Details: Deeply describe the main subjects, including specific colors, textures, \
materials, and any in-world text or numbers (signs, license plates, painted markings — \
ignore UI text).
2. Environment & Lighting: Describe the background, weather, lighting conditions (e.g., \
bright sunlight, overcast, neon), and surrounding elements.
3. Action & State: Describe what the subjects are doing in the space. When a playable \
character or person is visible (e.g. a third-person view), ALWAYS name their concrete action \
and locomotion with a specific verb — e.g. walking, running, sprinting, jumping, climbing, \
swimming, crouching, riding or driving, fighting or attacking, dodging or rolling, aiming, \
gathering, or standing idle — rather than only static positioning. Describe the character's \
OWN action; never describe how the camera follows or frames them.
4. Atmosphere/Mood: Conclude with a sentence capturing the emotional resonance, aesthetic, \
or overall vibe of the scene — match the mood to what is actually shown.

CRITICAL RULES:
- Describe ONLY the content of the scene. Do NOT describe the camera in any way: no shot \
types, no framing, no viewpoint, no camera movement. When a character is visible, describe \
them directly as the main subject, not how they are framed.
- Write a single flowing paragraph of roughly 3-6 sentences (about 60-120 words). No bullets.
- Start directly with the subject. No filler like "The video shows...".
- Describe it as live-action footage. Do NOT mention game engines, graphics, or gameplay.
- Output the caption and nothing else."""

from .. import config  # noqa: E402

DEFAULT_MODEL = config.CAPTION_MODEL   # default Qwen3.6-35B-A3B-FP8; alt Qwen3-VL-8B-Instruct


class QwenCaptioner(Captioner):
    def __init__(self, model=DEFAULT_MODEL, n_keyframes=12, max_pixels=401408, max_tokens=256):
        from vllm import LLM, SamplingParams

        self.n_keyframes = n_keyframes
        self.llm = LLM(model=model, trust_remote_code=True,
                       limit_mm_per_prompt={"image": n_keyframes}, enforce_eager=True,
                       max_num_seqs=32, max_model_len=16384,
                       mm_processor_kwargs={"max_pixels": max_pixels})
        self.sp = SamplingParams(temperature=0.3, max_tokens=max_tokens)

    def _msgs(self, frames):
        from PIL import Image
        idx = np.linspace(0, len(frames) - 1, min(self.n_keyframes, len(frames))).round().astype(int)
        imgs = [Image.fromarray(np.asarray(frames[i])[..., :3].astype("uint8")) for i in idx]
        content = [{"type": "image_pil", "image_pil": im} for im in imgs]
        content.append({"type": "text", "text": "Caption this clip."})
        return [{"role": "system", "content": PROMPT}, {"role": "user", "content": content}]

    def caption(self, frames) -> str:
        out = self.llm.chat([self._msgs(frames)], sampling_params=self.sp, use_tqdm=False,
                            chat_template_kwargs={"enable_thinking": False})  # 35B-A3B is a thinker
        return out[0].outputs[0].text.strip()
