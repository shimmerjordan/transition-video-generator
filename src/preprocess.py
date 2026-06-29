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


def active_rects(cleanup: dict, t: float) -> list:
    """返回在时刻 t 生效的去除矩形:全段(legacy watermarks/subtitles)+ 命中 t 的分段 regions。"""
    rects = list(cleanup.get("watermarks", []) or []) + list(cleanup.get("subtitles", []) or [])
    for reg in cleanup.get("regions", []) or []:
        rng = reg.get("range")
        if rng is None or (float(rng[0]) <= t <= float(rng[1])):
            rects += list(reg.get("watermarks", []) or []) + list(reg.get("subtitles", []) or [])
    return [list(map(int, r)) for r in rects]


def has_any_rects(cleanup: dict) -> bool:
    if cleanup.get("watermarks") or cleanup.get("subtitles"):
        return True
    return any((reg.get("watermarks") or reg.get("subtitles"))
               for reg in (cleanup.get("regions", []) or []))


def _inpaint_chunk(src: str, out: str, start: int, count: int,
                   cleanup: dict, fps: float, size: tuple) -> str:
    """处理 [start,start+count) 帧;按每帧时刻的生效矩形 inpaint(分段去水印),只算包围盒。"""
    w, h = size
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    cache: dict = {}

    def box_for(t):
        rects = active_rects(cleanup, t)
        key = tuple(map(tuple, sorted(rects)))
        if key not in cache:
            if not rects:
                cache[key] = None
            else:
                m = np.zeros((h, w), np.uint8)
                for x0, y0, x1, y1 in rects:
                    cv2.rectangle(m, (x0, y0), (x1, y1), 255, -1)
                m = cv2.dilate(m, kernel)
                ys, xs = np.where(m > 0)
                pad = 12
                bx0, bx1 = max(0, xs.min() - pad), min(w, xs.max() + pad + 1)
                by0, by1 = max(0, ys.min() - pad), min(h, ys.max() + pad + 1)
                cache[key] = (by0, by1, bx0, bx1, m[by0:by1, bx0:bx1])
        return cache[key]

    cap = cv2.VideoCapture(src)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    with video.FrameWriter(out, fps, (w, h)) as wr:
        for i in range(count):
            ok, fr = cap.read()
            if not ok:
                break
            b = box_for((start + i) / fps)
            if b:
                by0, by1, bx0, bx1, sub = b
                fr[by0:by1, bx0:bx1] = cv2.inpaint(fr[by0:by1, bx0:bx1], sub, 3, cv2.INPAINT_TELEA)
            wr.write(fr)
    cap.release()
    return out


def dewatermark_bg(cfg: dict, name: str, root: str, log=print) -> str:
    """对某背景去水印(支持按时间分段),产出 data/work/clean/<name>.mp4。无矩形则直接转存。"""
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
    if not has_any_rects(cleanup):
        log(f"[去水印] {name}: 无去除区域,直接转存")
        shutil.copyfile(src, out)
        return out

    info = video.video_info(src)
    fps = info["fps"] or get(cfg, "project.fps", 30)
    w, h = info["width"], info["height"]
    total = info["count"] or 0
    workers = min(os.cpu_count() or 4, 4)     # cv2 释放 GIL,多线程并行;限 4 路 NVENC 会话
    chunk = math.ceil(total / workers) if total else 0

    work_dir = os.path.join(contract.work_root(root), "_chunks", name)
    video.ensure_dir(work_dir)
    jobs, parts = [], []
    for i in range(workers):
        s = i * chunk
        if total and s >= total:
            break
        part = os.path.join(work_dir, f"part_{i:03d}.mp4")
        parts.append(part)
        jobs.append((src, part, s, chunk, cleanup, float(fps), (w, h)))
    nreg = len(cleanup.get("regions", []) or [])
    log(f"[去水印] {name}: {video.gpu_codec()} × {len(jobs)} 线程,共 {total} 帧"
        f"(全段矩形+{nreg} 个分段区间)…")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(lambda a: _inpaint_chunk(*a), jobs))

    lst = os.path.join(work_dir, "concat.txt")
    with open(lst, "w", encoding="utf-8") as f:
        for p in parts:
            f.write(f"file '{p.replace(os.sep, '/')}'\n")
    subprocess.run([video.ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "concat",
                    "-safe", "0", "-i", lst, "-c", "copy", out], check=True)
    shutil.rmtree(work_dir, ignore_errors=True)
    log(f"[去水印] {name}: 完成 {total} 帧 → {out}")
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
