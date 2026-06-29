"""S6 重打光(可选,★真实度核心):按背景光照参数对主体重新打光,使人景统一。

设计:用 IC-Light,以 S4 估计的背景光照(方位/色温/强度,见 plates/seg_*.light.json)
为条件,对(换装后的)主体重打光。这是"假/真"的分水岭,必须由背景真实光照驱动。

当前实现:IC-Light 需 torch + 权重。未就绪时本步**透传**(复制 S5 输出)。
为离线也有基础人景协调,提供一个轻量回退 `soft_match`(可选):向背景环境色做温和
偏移 —— 非物理打光,仅缓解色调割裂;真实打光仍以接 IC-Light 为准。

输入:data/work/garment/seg_{id}/ + plates/seg_{id}.light.json
输出:data/work/relit/seg_{id}/
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import video  # noqa: E402
from src.utils.config import get, load_config, resolve_path  # noqa: E402


def iclight_available(cfg: dict) -> tuple[bool, str]:
    try:
        import torch  # noqa: F401
    except Exception:
        return False, "未安装 torch"
    d = resolve_path(cfg, get(cfg, "models.ic_light", "models/ic-light"))
    if not os.path.isdir(d):
        return False, "缺少 IC-Light 权重(models/)"
    return True, "ok"


def relight_iclight(cfg, frames, alpha, light):  # pragma: no cover
    raise NotImplementedError(
        "IC-Light 待接入:以背景光照(方位/色温/强度)为条件对主体重打光。"
    )


def soft_match(frame: np.ndarray, alpha: np.ndarray, ambient_bgr, strength: float) -> np.ndarray:
    """轻量回退:仅对主体区域向背景环境色做温和色调偏移(非物理打光)。"""
    a = (alpha.astype(np.float32) / 255.0)[..., None] * strength
    target = np.array(ambient_bgr, dtype=np.float32).reshape(1, 1, 3)
    f = frame.astype(np.float32)
    cur_mean = f.reshape(-1, 3).mean(0).reshape(1, 1, 3)
    shifted = f + (target - cur_mean) * a
    return np.clip(shifted, 0, 255).astype(np.uint8)


def _passthrough(src_dir: str, out_dir: str) -> int:
    video.ensure_dir(out_dir)
    paths = sorted(glob.glob(os.path.join(src_dir, "f*.png")))
    for p in paths:
        shutil.copyfile(p, os.path.join(out_dir, os.path.basename(p)))
    return len(paths)


def process_segment(cfg: dict, sid: int, work_root: str, enabled: bool) -> dict:
    src_dir = os.path.join(work_root, "garment", f"seg_{sid}")
    out_dir = os.path.join(work_root, "relit", f"seg_{sid}")
    light_path = os.path.join(work_root, "plates", f"seg_{sid}.light.json")
    fallback = get(cfg, "relight.fallback", "passthrough")  # passthrough | soft_match
    mode = "passthrough"

    if not enabled:
        _passthrough(src_dir, out_dir)
    else:
        ok, why = iclight_available(cfg)
        if ok:
            try:
                # 真实路径(待接入)
                relight_iclight(cfg, None, None, None)
                mode = "iclight"
            except NotImplementedError as e:
                print(f"[s6][告警] {e}\n[s6] 段 {sid} 回退 {fallback}")
                ok = False
        if not ok:
            if fallback == "soft_match" and os.path.isfile(light_path):
                light = json.load(open(light_path, encoding="utf-8"))
                ambient = light.get("ambient_color_bgr", [128, 128, 128])
                strength = float(get(cfg, "relight.soft_strength", 0.25))
                video.ensure_dir(out_dir)
                src = sorted(glob.glob(os.path.join(src_dir, "f*.png")))
                adir = os.path.join(work_root, "alpha", f"seg_{sid}")
                for p in src:
                    name = os.path.basename(p)
                    frame = video.imread(p)
                    ap = os.path.join(adir, name)
                    alpha = video.imread(ap, cv2.IMREAD_GRAYSCALE) if os.path.isfile(ap) \
                        else np.full(frame.shape[:2], 255, np.uint8)
                    video.imwrite(os.path.join(out_dir, name),
                                  soft_match(frame, alpha, ambient, strength))
                mode = "soft_match"
            else:
                _passthrough(src_dir, out_dir)

    n = len(glob.glob(os.path.join(out_dir, "f*.png")))
    print(f"[s6] 段 {sid}: {n} 帧 → {out_dir}({mode})")
    return {"segment": sid, "frames": n, "relit_dir": out_dir, "mode": mode}


def run(config_path: str | None, only_segment: int | None) -> list[dict]:
    cfg = load_config(config_path)
    work_root = resolve_path(cfg, "data/work")
    enabled = bool(get(cfg, "project.enable_relight", True))
    g_root = os.path.join(work_root, "garment")
    if not os.path.isdir(g_root):
        raise SystemExit("缺少 data/work/garment,请先运行 S5")
    sids = sorted(int(os.path.basename(d).split("_")[1])
                  for d in glob.glob(os.path.join(g_root, "seg_*")))
    if only_segment is not None:
        sids = [s for s in sids if s == only_segment]
    results = [process_segment(cfg, s, work_root, enabled) for s in sids]
    print(f"[s6] 完成 {len(results)} 段(enable_relight={enabled})")
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="S6 重打光(可选)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--segment", type=int, default=None)
    args = ap.parse_args()
    run(args.config, args.segment)


if __name__ == "__main__":
    main()
