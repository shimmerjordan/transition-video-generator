"""MatAnyone 软 alpha 细化:用 SAM2 的首帧 mask 作目标,逐帧传播出发丝级软边 alpha。

模型/代码来自 HF(PeiqingYang/MatAnyone:权重 models/matanyone;matanyone2 包 models/matanyone_code)。
对外只暴露 refine_person(frames_bgr, mask0) -> [HxW uint8 soft alpha]。
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

_MODEL = None
_READY = False


def _root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def available() -> bool:
    r = _root()
    return (os.path.isfile(os.path.join(r, "models", "matanyone", "model.safetensors"))
            and os.path.isdir(os.path.join(r, "models", "matanyone_code", "matanyone2")))


def _load():
    """惰性加载 MatAnyone2 模型(GPU)。"""
    global _MODEL, _READY
    if _READY:
        return _MODEL
    r = _root()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    for p in (os.path.join(r, "models", "matanyone_code"),
              os.path.join(r, "models", "matanyone_code", "hugging_face")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import torch  # noqa
    from matanyone2.model.matanyone2 import MatAnyone2
    _MODEL = MatAnyone2.from_pretrained(os.path.join(r, "models", "matanyone")).to("cuda").eval()
    _READY = True
    return _MODEL


def refine_person(frames_bgr: list[np.ndarray], mask0: np.ndarray,
                  r_erode: int = 10, r_dilate: int = 10, n_warmup: int = 10) -> list[np.ndarray]:
    """frames_bgr: BGR 帧序列;mask0: 首帧二值 mask(HxW uint8)。返回每帧软 alpha(HxW uint8)。"""
    model = _load()
    from matanyone2.inference.inference_core import InferenceCore
    from matanyone2_wrapper import matanyone2
    proc = InferenceCore(model, cfg=model.cfg)        # 每个目标用独立 processor(清空记忆)
    frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr]
    _, phas = matanyone2(proc, frames_rgb, mask0.astype(np.uint8),
                         r_erode=r_erode, r_dilate=r_dilate, n_warmup=n_warmup)
    return [p[..., 0] if p.ndim == 3 else p for p in phas]
