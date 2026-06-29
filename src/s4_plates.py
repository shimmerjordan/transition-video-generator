"""S4 背景锁定平面:把每段指定的背景素材稳定成 locked plate,与锁定域前景对齐;
并从平面估计主光方向/色温,传给 S6/S7。

- 背景素材自身的运镜用 S2 的跟踪器消除 → 近似"锁定机位"平面。
- 平面尺寸 = 锁定域帧尺寸(w,h);S8 贴回运镜后再 center-crop 到输出尺寸(overscan 余量)。
- 素材帧数不足时 ping-pong 循环,避免跳切。

输出:
    data/work/plates/seg_{id}/f{idx:05d}.png   锁定背景平面帧
    data/work/plates/seg_{id}.light.json        光照估计(可被 config 覆盖)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import video  # noqa: E402
from src.utils.config import get, load_config, resolve_path  # noqa: E402
from src.s2_camera import stabilize, track_segment  # noqa: E402


def fit_cover(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """等比缩放并中心裁剪到目标尺寸(cover)。"""
    tw, th = size
    h, w = img.shape[:2]
    scale = max(tw / w, th / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    x0 = (nw - tw) // 2
    y0 = (nh - th) // 2
    return r[y0:y0 + th, x0:x0 + tw]


def pingpong(frames: list, n: int) -> list:
    """把 frames 延展/循环到 n 帧(往返,避免硬跳)。"""
    if not frames:
        raise SystemExit("背景素材读到 0 帧")
    if len(frames) >= n:
        return frames[:n]
    seq = frames + frames[-2:0:-1]  # 往返
    out = []
    i = 0
    while len(out) < n:
        out.append(seq[i % len(seq)])
        i += 1
    return out


def estimate_light(plate: np.ndarray) -> dict:
    """从背景平面粗估主光:高光区域颜色(色温)与方位。"""
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    thr = np.percentile(gray, 90)
    ys, xs = np.where(gray >= thr)
    if len(xs) == 0:
        cx, cy = w / 2, h / 2
        key_bgr = plate.reshape(-1, 3).mean(0)
    else:
        cx, cy = float(xs.mean()), float(ys.mean())
        key_bgr = plate[ys, xs].mean(0)
    ambient_bgr = plate.reshape(-1, 3).mean(0)
    # 方位:从画面中心指向高光质心
    azimuth = float(np.degrees(np.arctan2(cy - h / 2, cx - w / 2)))
    elevation = float(90.0 * (1.0 - cy / h))  # 越靠上 → 越高
    b, g, r = key_bgr
    color_temp_hint = "warm" if r > b else "cool"
    return {
        "key_color_bgr": [float(x) for x in key_bgr],
        "ambient_color_bgr": [float(x) for x in ambient_bgr],
        "azimuth_deg": round(azimuth, 1),
        "elevation_deg": round(elevation, 1),
        "intensity": round(float(thr) / 255.0, 3),
        "color_temp_hint": color_temp_hint,
    }


def _count_locked(work_root: str, sid: int) -> int:
    d = os.path.join(work_root, "locked", f"seg_{sid}")
    n = len(glob.glob(os.path.join(d, "f*.png")))
    if n == 0:
        raise SystemExit(f"段 {sid} 缺少锁定域帧,请先运行 S2")
    return n


def _seg_cfg(cfg: dict, sid: int) -> dict:
    for s in get(cfg, "segments", []) or []:
        if s.get("id") == sid:
            return s
    raise SystemExit(f"config.segments 中找不到段 {sid} 的背景配置")


def process_segment(cfg: dict, sid: int, work_root: str) -> dict:
    n = _count_locked(work_root, sid)
    first = video.imread(os.path.join(work_root, "locked", f"seg_{sid}", "f00000.png"))
    h, w = first.shape[:2]

    scfg = _seg_cfg(cfg, sid)
    bg_dir = resolve_path(cfg, get(cfg, "input.backgrounds_dir", "data/input/backgrounds"))
    bg_path = os.path.join(bg_dir, os.path.basename(scfg["background"]))
    if not os.path.isfile(bg_path):
        raise SystemExit(f"段 {sid} 背景素材不存在:{bg_path}")

    info = video.video_info(bg_path)
    fps = info["fps"] or get(cfg, "project.fps", 30)
    trim = scfg.get("bg_trim", [0.0, None]) or [0.0, None]
    start_f = int(round(float(trim[0]) * fps)) if trim[0] else 0
    raw = list(video.read_frames(bg_path, start=start_f, count=None))
    if trim[1]:
        end_f = int(round(float(trim[1]) * fps)) - start_f
        raw = raw[:max(1, end_f)]

    # 消除背景自身运镜 → 锁定平面
    transforms = track_segment(raw)
    locked_bg = stabilize(raw, transforms) if len(raw) > 1 else raw
    locked_bg = [fit_cover(f, (w, h)) for f in locked_bg]
    plates = pingpong(locked_bg, n)

    out_dir = os.path.join(work_root, "plates", f"seg_{sid}")
    video.ensure_dir(out_dir)
    for i, f in enumerate(plates):
        video.imwrite(os.path.join(out_dir, f"f{i:05d}.png"), f)

    # 光照:配置覆盖优先
    light = estimate_light(plates[0])
    lc = scfg.get("light", {}) or {}
    if lc.get("direction") not in (None, "auto"):
        d = lc["direction"]
        light["azimuth_deg"], light["elevation_deg"] = float(d[0]), float(d[1])
        light["source"] = "config"
    if lc.get("color_temp") not in (None, "auto"):
        light["color_temp_kelvin"] = float(lc["color_temp"])
    if lc.get("intensity") not in (None, "auto"):
        light["intensity"] = float(lc["intensity"])
    with open(os.path.join(work_root, "plates", f"seg_{sid}.light.json"),
              "w", encoding="utf-8") as f:
        json.dump(light, f, ensure_ascii=False, indent=2)

    print(f"[s4] 段 {sid}: {len(plates)} 帧 plate ← {os.path.basename(bg_path)}"
          f"(方位 {light['azimuth_deg']}°,{light['color_temp_hint']})")
    return {"segment": sid, "frames": len(plates), "plate_dir": out_dir, "light": light}


def run(config_path: str | None, only_segment: int | None) -> list[dict]:
    cfg = load_config(config_path)
    work_root = resolve_path(cfg, "data/work")
    locked_root = os.path.join(work_root, "locked")
    if not os.path.isdir(locked_root):
        raise SystemExit("缺少 data/work/locked,请先运行 S2")
    sids = sorted(int(os.path.basename(d).split("_")[1])
                  for d in glob.glob(os.path.join(locked_root, "seg_*")))
    if only_segment is not None:
        sids = [s for s in sids if s == only_segment]
    results = [process_segment(cfg, s, work_root) for s in sids]
    print(f"[s4] 完成 {len(results)} 段")
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="S4 背景锁定平面 + 光照估计")
    ap.add_argument("--config", default=None)
    ap.add_argument("--segment", type=int, default=None)
    args = ap.parse_args()
    run(args.config, args.segment)


if __name__ == "__main__":
    main()
