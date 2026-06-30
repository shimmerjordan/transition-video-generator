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
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np

from src import contract
from src.utils import video
from src.utils.config import get, resolve_path


def _bg_file(cfg: dict, bg: dict) -> str:
    bg_dir = resolve_path(cfg, get(cfg, "input.backgrounds_dir", "data/input/backgrounds"))
    return os.path.join(bg_dir, os.path.basename(bg["file"]))


def active_rects(cleanup: dict, t: float) -> list:
    """返回在时刻 t 生效的去除矩形:marks(每印记一轨,带 range)+ 兼容旧 regions / 全段。"""
    rects = list(cleanup.get("watermarks", []) or []) + list(cleanup.get("subtitles", []) or [])
    for reg in cleanup.get("regions", []) or []:
        rng = reg.get("range")
        if rng is None or (float(rng[0]) <= t <= float(rng[1])):
            rects += list(reg.get("watermarks", []) or []) + list(reg.get("subtitles", []) or [])
    for mk in cleanup.get("marks", []) or []:
        rng = mk.get("range")
        if mk.get("rect") and (rng is None or (float(rng[0]) <= t <= float(rng[1]))):
            rects.append(mk["rect"])
    return [list(map(int, r)) for r in rects]


def has_any_rects(cleanup: dict) -> bool:
    if cleanup.get("watermarks") or cleanup.get("subtitles"):
        return True
    if any(m.get("rect") for m in (cleanup.get("marks", []) or [])):
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


def dewatermark_bg(cfg: dict, name: str, root: str, log=print, progress=None) -> str:
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

    if get(cfg, "dewatermark.engine", "cv2") == "sd":
        return _sd_inpaint_pass(cfg, name, src, out, cleanup, root, log, progress)

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
        futs = [ex.submit(_inpaint_chunk, *a) for a in jobs]
        done = 0
        for f in as_completed(futs):
            f.result()
            done += 1
            if progress:
                progress(done / len(futs))

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


_SD_PIPE = None


def _sd_pipe(cfg: dict):
    """构建/缓存 SD inpaint 管线(GPU,fp16)。"""
    global _SD_PIPE
    if _SD_PIPE is None:
        import torch
        from diffusers import AutoPipelineForInpainting
        model = get(cfg, "dewatermark.model", "Lykon/dreamshaper-8-inpainting")
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        try:
            pipe = AutoPipelineForInpainting.from_pretrained(
                model, torch_dtype=torch.float16, variant="fp16", safety_checker=None)
        except Exception:
            pipe = AutoPipelineForInpainting.from_pretrained(
                model, torch_dtype=torch.float16, safety_checker=None)
        pipe = pipe.to("cuda")
        pipe.set_progress_bar_config(disable=True)
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass
        _SD_PIPE = pipe
    return _SD_PIPE


def _sd_inpaint_pass(cfg, name, src, out, cleanup, root, log, progress) -> str:
    """逐帧 SD 扩散修复(GPU):每个生效矩形单独裁剪→修复→贴回。质量好、慢、吃 GPU。"""
    import torch
    from PIL import Image
    pipe = _sd_pipe(cfg)
    seed = int(get(cfg, "dewatermark.seed", 0))
    S = int(get(cfg, "dewatermark.size", 512))
    steps = int(get(cfg, "dewatermark.steps", 25))
    gs = float(get(cfg, "dewatermark.guidance", 8))
    prompt = get(cfg, "dewatermark.prompt", "clean background, seamless, photorealistic")
    neg = get(cfg, "dewatermark.negative", "text, watermark, logo, letters, characters, words")

    info = video.video_info(src)
    fps, w, h = info["fps"] or get(cfg, "project.fps", 30), info["width"], info["height"]
    total = info["count"] or 0
    log(f"[去水印·SD] {name}: {get(cfg,'dewatermark.model','dreamshaper-8-inpainting')} GPU,共 {total} 帧(慢)…")

    n = 0
    with video.FrameWriter(out, fps, (w, h)) as wr:
        for fr in video.read_frames(src):
            t = n / fps
            for x0, y0, x1, y1 in active_rects(cleanup, t):
                cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
                half = max(x1 - x0, y1 - y0) // 2 + 60
                bx0, by0 = max(0, cx - half), max(0, cy - half)
                bx1, by1 = min(w, cx + half), min(h, cy + half)
                crop = fr[by0:by1, bx0:bx1].copy()
                m = np.zeros(crop.shape[:2], np.uint8)
                cv2.rectangle(m, (x0 - bx0, y0 - by0), (x1 - bx0, y1 - by0), 255, -1)
                m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
                ci = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).resize((S, S))
                mi = Image.fromarray(m).resize((S, S))
                gen = torch.Generator("cuda").manual_seed(seed)  # 固定噪声→减少逐帧抖动
                res = pipe(prompt=prompt, negative_prompt=neg, image=ci, mask_image=mi,
                           num_inference_steps=steps, guidance_scale=gs, generator=gen).images[0]
                res = cv2.cvtColor(np.array(res.resize((bx1 - bx0, by1 - by0))), cv2.COLOR_RGB2BGR)
                mm = (cv2.resize(m, (bx1 - bx0, by1 - by0)) > 127)[..., None]
                fr[by0:by1, bx0:bx1] = np.where(mm, res, crop)
            wr.write(fr)
            n += 1
            if progress and total:
                progress(n / total)
            if n % 50 == 0:
                log(f"[去水印·SD] {name}: {n}/{total} 帧…")
    log(f"[去水印·SD] {name}: 完成 {n} 帧 → {out}")
    return out


def run_dewatermark(cfg: dict, root: str, name: str | None, log=print, progress=None) -> None:
    names = [name] if name else list((get(cfg, "backgrounds", {}) or {}).keys())
    for i, nm in enumerate(names):
        dewatermark_bg(cfg, nm, root, log,
                       progress=(lambda f, i=i: progress((i + f) / len(names))) if progress else None)


def run_clips(cfg: dict, root: str, name: str | None, log=print) -> None:
    names = [name] if name else list((get(cfg, "backgrounds", {}) or {}).keys())
    for nm in names:
        make_clips(cfg, nm, root, log)
