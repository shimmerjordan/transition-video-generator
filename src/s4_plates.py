"""S4 背景锁定平面:解析每段引用的背景子片段 → 去元素(本地静态 inpaint)→
消除自身运镜 → fit 到锁定尺寸 → 按地面策略处理 → 估计光照。

新配置(v2):
    backgrounds.<name>.clips[].{id,range}   命名子片段(秒)
    backgrounds.<name>.cleanup.{watermarks,subtitles,movers}  静态去元素 mask 矩形
    segments[*].background_clip              引用子片段 id
    segments[*].ground                       as_is | generate | virtual_plane

去路人(movers)等动态元素建议走 provider=product(video-inpaint);本地仅处理静态水印/字幕。
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

from src import contract  # noqa: E402
from src.utils import video  # noqa: E402
from src.utils.config import get, load_config, resolve_path  # noqa: E402
from src.s2_camera import motion_magnitude, stabilize, track_segment  # noqa: E402


def fit_cover(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    tw, th = size
    h, w = img.shape[:2]
    scale = max(tw / w, th / h)
    # 缩小用 INTER_AREA(抗锯齿、最锐),放大用 LANCZOS4(避免发虚)
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LANCZOS4
    r = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))), interpolation=interp)
    x0 = (r.shape[1] - tw) // 2
    y0 = (r.shape[0] - th) // 2
    return r[y0:y0 + th, x0:x0 + tw]


def pingpong(frames: list, n: int) -> list:
    if not frames:
        raise SystemExit("背景素材读到 0 帧")
    if len(frames) >= n:
        return frames[:n]
    seq = frames + frames[-2:0:-1]
    return [seq[i % len(seq)] for i in range(n)]


def resolve_clip(cfg: dict, clip_id: str) -> tuple[str, list, dict]:
    """根据 segment.background_clip 找到 (背景文件, [起止秒], cleanup)。"""
    bg_dir = resolve_path(cfg, get(cfg, "input.backgrounds_dir", "data/input/backgrounds"))
    for name, bg in (get(cfg, "backgrounds", {}) or {}).items():
        for clip in bg.get("clips", []):
            if clip.get("id") == clip_id:
                return (os.path.join(bg_dir, os.path.basename(bg["file"])),
                        clip.get("range", [0, None]), bg.get("cleanup", {}) or {})
    raise SystemExit(f"backgrounds 中找不到子片段:{clip_id}")


def build_cleanup_mask(shape, cleanup: dict) -> np.ndarray | None:
    """由静态矩形(水印/字幕)生成 inpaint mask。movers 不在此(交给 product)。"""
    h, w = shape[:2]
    mask = np.zeros((h, w), np.uint8)
    rects = (cleanup.get("watermarks", []) or []) + (cleanup.get("subtitles", []) or [])
    for x0, y0, x1, y1 in rects:
        cv2.rectangle(mask, (int(x0), int(y0)), (int(x1), int(y1)), 255, -1)
    if not rects:
        return None
    return cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))


def apply_ground(plate: np.ndarray, strategy: str) -> np.ndarray:
    """地面策略。as_is 原样;virtual_plane 在下方铺一层简易地面(占位,供接触阴影落地);
    generate 需走生成式(provider),此处回退 as_is 并由调用方告警。"""
    if strategy != "virtual_plane":
        return plate
    h, w = plate.shape[:2]
    out = plate.copy()
    y0 = int(h * 0.62)
    ground = np.zeros((h - y0, w, 3), np.uint8)
    top = np.array([120, 130, 140], np.float32)   # 远处地面
    bot = np.array([60, 75, 95], np.float32)       # 近处地面(BGR)
    for i in range(ground.shape[0]):
        t = i / max(1, ground.shape[0] - 1)
        ground[i, :] = (top * (1 - t) + bot * t).astype(np.uint8)
    blend = 0.85
    out[y0:] = (plate[y0:].astype(np.float32) * (1 - blend) + ground.astype(np.float32) * blend).astype(np.uint8)
    return out


def estimate_light(plate: np.ndarray) -> dict:
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    thr = np.percentile(gray, 90)
    ys, xs = np.where(gray >= thr)
    if len(xs) == 0:
        cx, cy, key = w / 2, h / 2, plate.reshape(-1, 3).mean(0)
    else:
        cx, cy, key = float(xs.mean()), float(ys.mean()), plate[ys, xs].mean(0)
    b, g, r = key
    return {
        "key_color_bgr": [float(x) for x in key],
        "ambient_color_bgr": [float(x) for x in plate.reshape(-1, 3).mean(0)],
        "azimuth_deg": round(float(np.degrees(np.arctan2(cy - h / 2, cx - w / 2))), 1),
        "elevation_deg": round(float(90.0 * (1.0 - cy / h)), 1),
        "intensity": round(float(thr) / 255.0, 3),
        "color_temp_hint": "warm" if r > b else "cool",
    }


def _count_locked(work_root: str, sid: int) -> tuple[int, tuple[int, int]]:
    d = os.path.join(work_root, "locked", f"seg_{sid}")
    fs = sorted(glob.glob(os.path.join(d, "f*.png")))
    if not fs:
        raise SystemExit(f"段 {sid} 缺少锁定域帧,请先运行 S2")
    img = video.imread(fs[0])
    return len(fs), (img.shape[1], img.shape[0])


def _seg_cfg(cfg: dict, sid: int) -> dict:
    for s in get(cfg, "segments", []) or []:
        if s.get("id") == sid:
            return s
    raise SystemExit(f"config.segments 中找不到段 {sid}")


def clip_plate(cfg: dict, clip_id: str, n: int, size: tuple, root: str,
               ground: str = "as_is") -> list:
    """把某背景片段做成 n 帧锁定平面(读片段/原片→去自身运镜→fit→地面→pingpong 到 n 帧)。"""
    w, h = size
    clip_video = contract.clip_path(root, clip_id)
    if os.path.isfile(clip_video):
        raw = list(video.read_frames(clip_video))
    else:
        bg_path, rng, cleanup = resolve_clip(cfg, clip_id)
        if not os.path.isfile(bg_path):
            raise SystemExit(f"背景文件不存在:{bg_path}")
        info = video.video_info(bg_path)
        fps = info["fps"] or get(cfg, "project.fps", 30)
        start_f = int(round(float(rng[0]) * fps)) if rng and rng[0] else 0
        raw = list(video.read_frames(bg_path, start=start_f, count=None))
        if rng and len(rng) > 1 and rng[1]:
            raw = raw[:max(1, int(round(float(rng[1]) * fps)) - start_f)]
        mask = build_cleanup_mask(raw[0].shape, cleanup) if raw else None
        if mask is not None:
            raw = [cv2.inpaint(f, mask, 3, cv2.INPAINT_TELEA) for f in raw]
    if not raw:
        raise SystemExit(f"片段 {clip_id} 读到 0 帧")
    # 背景自身运镜:只有确有明显运动才做反稳定(warpAffine 会重采样变糊);
    # 近静止的背景跳过稳定化以保留原始清晰度。
    if len(raw) > 1:
        transforms = track_segment(raw)
        disp = motion_magnitude(transforms, (raw[0].shape[1], raw[0].shape[0]))
        if disp < 2.5:
            print(f"[s4] {clip_id}: 背景运镜仅 {disp:.1f}px,跳过稳定化(保清晰)")
            locked_bg = raw
        else:
            print(f"[s4] {clip_id}: 背景运镜 {disp:.1f}px,做反稳定锁定")
            locked_bg = stabilize(raw, transforms)
    else:
        locked_bg = raw
    if ground == "generate":
        print(f"[s4] {clip_id}: ground=generate 需生成式(provider),本地回退 as_is")
    locked_bg = [apply_ground(fit_cover(f, (w, h)), ground) for f in locked_bg]
    return pingpong(locked_bg, n)


def _write_plate(cfg, work_root, sid, plates, w, h, clip_id, ground, scfg):
    out_dir = contract.seg_dir(cfg.get("_root", "."), "plates", sid)
    video.ensure_dir(out_dir)
    for i, f in enumerate(plates):
        video.imwrite(os.path.join(out_dir, f"f{i:05d}.png"), f)
    contract.write_manifest(out_dir, step="plates", segment=sid, kind="frames",
                            fps=float(get(cfg, "project.fps", 30)), width=w, height=h,
                            count=len(plates), background_clip=clip_id, ground=ground)
    light = estimate_light(plates[0])
    lc = (scfg or {}).get("light", {}) or {}
    if lc.get("direction") not in (None, "auto"):
        light["azimuth_deg"], light["elevation_deg"] = float(lc["direction"][0]), float(lc["direction"][1])
    if lc.get("color_temp") not in (None, "auto"):
        light["color_temp_kelvin"] = float(lc["color_temp"])
    if lc.get("intensity") not in (None, "auto"):
        light["intensity"] = float(lc["intensity"])
    with open(os.path.join(work_root, "plates", f"seg_{sid}.light.json"), "w", encoding="utf-8") as f:
        json.dump(light, f, ensure_ascii=False, indent=2)
    return out_dir


def build_timeline_plate(cfg: dict, sid: int, work_root: str, span_start: float,
                         segments: list, log=print) -> int:
    """★整段连续抠像配套:为段 sid 的全部锁定帧,按各背景时间区间切换,拼出一条整长 plate。"""
    n, (w, h) = _count_locked(work_root, sid)
    fps = float(get(cfg, "project.fps", 30))
    root = cfg.get("_root", ".")
    plate = [None] * n
    for seg in segments:
        clip = seg.get("background_clip")
        t = seg.get("time")
        if not clip or not t:
            continue
        i0 = max(0, int(round((float(t[0]) - span_start) * fps)))
        i1 = min(n, int(round((float(t[1]) - span_start) * fps)))
        cnt = i1 - i0
        if cnt <= 0:
            continue
        frames = clip_plate(cfg, clip, cnt, (w, h), root, seg.get("ground", "as_is"))
        for j in range(cnt):
            plate[i0 + j] = frames[j]
        log(f"[s4] 背景切换 [{t[0]}~{t[1]}s] ← {clip}({cnt} 帧)")
    # 填补未覆盖帧(前向/后向最近)
    last = None
    for i in range(n):
        if plate[i] is None:
            plate[i] = last
        else:
            last = plate[i]
    nxt = None
    for i in range(n - 1, -1, -1):
        if plate[i] is None:
            plate[i] = nxt if nxt is not None else np.zeros((h, w, 3), np.uint8)
        else:
            nxt = plate[i]
    _write_plate(cfg, work_root, sid, plate, w, h, "timeline", "as_is", None)
    log(f"[s4] 段 {sid}: 整长 plate {n} 帧(按时间线切换背景)")
    return n


def process_segment(cfg: dict, sid: int, work_root: str) -> dict:
    n, (w, h) = _count_locked(work_root, sid)
    scfg = _seg_cfg(cfg, sid)
    clip_id = scfg.get("background_clip")
    if not clip_id:
        raise SystemExit(f"段 {sid} 未配置 background_clip")
    ground = scfg.get("ground", "as_is")
    plates = clip_plate(cfg, clip_id, n, (w, h), cfg.get("_root", "."), ground)
    out_dir = _write_plate(cfg, work_root, sid, plates, w, h, clip_id, ground, scfg)
    print(f"[s4] 段 {sid}: {len(plates)} 帧 plate ← {clip_id}(ground={ground})")
    return {"segment": sid, "frames": len(plates), "plate_dir": out_dir, "clip": clip_id}


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
    ap = argparse.ArgumentParser(description="S4 背景锁定平面(子片段+去元素+地面)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--segment", type=int, default=None)
    args = ap.parse_args()
    run(args.config, args.segment)


if __name__ == "__main__":
    main()
