"""External paths + model ids, overridable by env (defaults match the validation box)."""
import os

WFORGE_SRC = os.environ.get("GAMEPIPE_WFORGE_SRC", "/root/nas/bigdata1/cjw/projs/datapipe/src")
TRANSNET_PKG = os.environ.get(
    "GAMEPIPE_TRANSNET_PKG", "/home/chijw/workspace/projs/WBench/third_party/transnetv2_pytorch")
HF = os.environ.get("GAMEPIPE_HF", "/root/nas/bigdata1/huggingface")
CAPTION_MODEL = os.environ.get("GAMEPIPE_CAPTION_MODEL", f"{HF}/Qwen3.6-35B-A3B-FP8")
HUD_MODEL = os.environ.get("GAMEPIPE_HUD_MODEL", f"{HF}/Qwen3-VL-8B-Instruct")
