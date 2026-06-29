"""预处理工具:去水印(整段) 与 片段裁剪。每个都产出标准视频件,供下游消费。

标准产物:
    去水印:data/work/clean/<bg>.mp4   (按 cleanup 矩形对整段做 inpaint)
    裁剪  :data/work/clips/<clipid>.mp4(从 clean 优先、否则原片 裁出每个子片段)

provider:
    cleanup=local   → cv2.inpaint 抹除静态水印/字幕(本实现)
    cleanup=product → 走 handoff(动态路人/复杂修复用付费视频修复)
"""
from __future__ import annotations

import math
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from src import contract
from src.utils import video
from src.utils.config import get, resolve_path


def _bg_file(cfg: dict, bg: dict) -> str:
    bg_dir = resolve_path(cfg, get(cfg, "input.backgrounds_dir", "data/input/backgrounds"))
    return os.path.join(bg_dir, os.path.basename(bg["file"]))


def _cleanup_mask(shape, cleanup: dict):
    h, w = shape[:2]
    rects = (cleanup.get("watermarks", []) or []) + (cleanup.get("subtitles", []) or [])
    if not rects:
        return None
    mask = np.zeros((h, w), np.uint8)
    for x0, y0, x1, y1 in rects:
        cv2.rectangle(mask, (int(x0), int(y0)), (int(x1), int(y1)), 255, -1)
    return cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))


def _rects_of(cleanup: dict) -> list:
    return [list(map(int, r)) for r in
            (cleanup.get("watermarks", []) or []) + (cleanup.get("subtitles", []) or [])]


def _inpaint_chunk(src: str, out: str, start: int, count: int,
                   rects: list, fps: float, size: tuple) -> str:
    """子进程:处理 [start,start+count) 帧,只在矩形包围盒内 inpaint,编码为 chunk。"""
    w, h = size
    mask = np.zeros((h, w), np.uint8)
    for x0, y0, x1, y1 in rects:
        cv2.rectangle(mask, (x0, y0), (x1, y1), 255, -1)
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    ys, xs = np.where(mask > 0)
    pad = 12
    bx0, bx1 = max(0, xs.min() - pad), min(w, xs.max() + pad + 1)
    by0, by1 = max(0, ys.min() - pad), min(h, ys.max() + pad + 1)
    sub = mask[by0:by1, bx0:bx1]
    cap = cv2.VideoCapture(src)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    with video.FrameWriter(out, fps, (w, h)) as wr:
        for _ in range(count):
            ok, fr = cap.read()
            if not ok:
                break
            fr[by0:by1, bx0:bx1] = cv2.inpaint(fr[by0:by1, bx0:bx1], sub, 3, cv2.INPAINT_TELEA)
            wr.write(fr)
    cap.release()
    return out


def dewatermark_bg(cfg: dict, name: str, root: str, log=print) -> str:
    """对某背景整段去水印,产出 data/work/clean/<name>.mp4。无矩形则直接转存。"""
    bg = (get(cfg, "backgrounds", {}) or {}).get(name)
    if not bg:
        raise SystemExit(f"backgrounds 无 {name}")
    src = _bg_file(cfg, bg)
    if not os.path.isfile(src):
        raise SystemExit(f"背景文件不存在:{src}")
    out = contract.clean_path(root, name)
    video.ensure_dir(os.path.dirname(out))

    cleanup = bg.get("cleanup", {}) or {}
    if cleanup.get("movers"):
        log(f"[去水印] {name}: 含 movers(动态路人),本地仅抹静态;动态请用 cleanup=product")

    info = video.video_info(src)
    fps, w, h = info["fps"] or get(cfg, "project.fps", 30), info["width"], info["height"]
    first = next(iter(video.read_frames(src, 0, 1)), None)
    if first is None:
        raise SystemExit("读不到帧")
    mask = _cleanup_mask(first.shape, cleanup)
    if mask is None:
        log(f"[去水印] {name}: 无去除区域,直接转存")
        shutil.copyfile(src, out)
        return out

    # 多进程分块并行(每块 GPU 编码),充分利用多核 + NVENC
    total = info["count"] or 0
    rects = _rects_of(cleanup)
    workers = min(os.cpu_count() or 4, 4)   # cv2 释放 GIL,多线程并行;限 4 路 NVENC 会话
    if total <= 0 or workers <= 1:
        return _dewatermark_serial(src, out, mask, fps, (w, h), log, name)

    chunk = math.ceil(total / workers)
    work_dir = os.path.join(contract.work_root(root), "_chunks", name)
    video.ensure_dir(work_dir)
    jobs, parts = [], []
    for i in range(workers):
        start = i * chunk
        if start >= total:
            break
        part = os.path.join(work_dir, f"part_{i:03d}.mp4")
        parts.append(part)
        jobs.append((src, part, start, chunk, rects, float(fps), (w, h)))
    log(f"[去水印] {name}: {video.gpu_codec()} × {len(jobs)} 线程并行,共 {total} 帧…")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(_dewatermark_chunk_star, jobs))

    # 拼接各块
    lst = os.path.join(work_dir, "concat.txt")
    with open(lst, "w", encoding="utf-8") as f:
        for p in parts:
            f.write(f"file '{p.replace(os.sep, '/')}'\n")
    subprocess.run([video.ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "concat",
                    "-safe", "0", "-i", lst, "-c", "copy", out], check=True)
    shutil.rmtree(work_dir, ignore_errors=True)
    log(f"[去水印] {name}: 完成 {total} 帧 → {out}")
    return out


def _dewatermark_chunk_star(args):
    return _inpaint_chunk(*args)


def _dewatermark_serial(src, out, mask, fps, size, log, name) -> str:
    w, h = size
    ys, xs = np.where(mask > 0)
    pad = 12
    x0, x1 = max(0, xs.min() - pad), min(w, xs.max() + pad + 1)
    y0, y1 = max(0, ys.min() - pad), min(h, ys.max() + pad + 1)
    sub = mask[y0:y1, x0:x1]
    with video.FrameWriter(out, fps, (w, h)) as wr:
        for fr in video.read_frames(src):
            fr[y0:y1, x0:x1] = cv2.inpaint(fr[y0:y1, x0:x1], sub, 3, cv2.INPAINT_TELEA)
            wr.write(fr)
    log(f"[去水印] {name}: 完成(单进程) → {out}")
    return out


def make_clips(cfg: dict, name: str, root: str, log=print) -> list[str]:
    """把某背景的所有子片段裁剪成独立视频(优先用 clean 版本)。"""
    bg = (get(cfg, "backgrounds", {}) or {}).get(name)
    if not bg:
        raise SystemExit(f"backgrounds 无 {name}")
    clean = contract.clean_path(root, name)
    src = clean if os.path.isfile(clean) else _bg_file(cfg, bg)
    used = "clean" if src == clean else "raw"
    if not os.path.isfile(src):
        raise SystemExit(f"源不存在:{src}")
    outs = []
    for clip in bg.get("clips", []):
        cid, rng = clip.get("id"), clip.get("range", [0, None])
        if not cid:
            continue
        out = contract.clip_path(root, cid)
        video.trim(src, out, float(rng[0] or 0), float(rng[1]) if len(rng) > 1 and rng[1] else None)
        outs.append(out)
        log(f"[裁剪] {cid}: {rng} ({used}) → {out}")
    log(f"[裁剪] {name}: 生成 {len(outs)} 个片段")
    return outs


def run_dewatermark(cfg: dict, root: str, name: str | None, log=print) -> None:
    names = [name] if name else list((get(cfg, "backgrounds", {}) or {}).keys())
    for nm in names:
        dewatermark_bg(cfg, nm, root, log)


def run_clips(cfg: dict, root: str, name: str | None, log=print) -> None:
    names = [name] if name else list((get(cfg, "backgrounds", {}) or {}).keys())
    for nm in names:
        make_clips(cfg, nm, root, log)
