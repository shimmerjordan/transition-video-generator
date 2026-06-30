"""S8 贴回运镜 + 拼接出片:

1) 把每段锁定域合成结果按 A_t 贴回源片运镜(前景背景一起 warp)→ 前景背景同运镜;
2) center-crop 掉 overscan 余量并缩放到输出分辨率;
3) 按 beats 在节拍点用指定转场拼接所有段;
4) 混入音乐导出成片。

转场(config.project.transition):
    hard_cut  : 帧精确硬切(默认,音画严格同步)
    crossfade : 段首数帧从上一段末帧溶解过来(保长度、不破坏同步)
    mask_wipe : 暂回退 crossfade

输入:data/work/comp_locked/seg_*/ + camera/seg_*.json + beats.json + 音乐
输出:data/output/final.mp4
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import video  # noqa: E402
from src.utils.config import get, load_config, resolve_path  # noqa: E402


def restore_segment_frames(work_root: str, sid: int, overscan: float,
                           out_size: tuple[int, int]) -> list[np.ndarray]:
    comp_dir = os.path.join(work_root, "comp_locked", f"seg_{sid}")
    names = sorted(glob.glob(os.path.join(comp_dir, "f*.png")))
    if not names:
        raise SystemExit(f"段 {sid} 缺少合成帧,请先运行 S7")
    cam_path = os.path.join(work_root, "camera", f"seg_{sid}.json")
    transforms, meta = ([], {})
    if os.path.isfile(cam_path):
        transforms, meta = video.load_transforms(cam_path)

    frames = [video.imread(p) for p in names]
    h, w = frames[0].shape[:2]

    # 是否有真实运镜:全部为单位阵(locked)则不贴回、不裁 overscan、不缩放 → 零拉伸变形
    ident = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64)
    has_motion = any(i < len(transforms) and not np.allclose(transforms[i], ident)
                     for i in range(len(frames)))
    if not has_motion:
        if (w, h) == tuple(out_size):
            return frames  # 三脚架:像素级原样,只换了背景
        return [cv2.resize(f, out_size) for f in frames]

    # 有运镜:贴回 + center-crop overscan + 缩放到输出
    mx, my = int(w * overscan / 2), int(h * overscan / 2)
    cx0, cy0, cx1, cy1 = mx, my, w - mx, h - my
    restored = []
    for i, f in enumerate(frames):
        if i < len(transforms) and not np.allclose(transforms[i], ident):
            f = cv2.warpAffine(f, transforms[i].astype(np.float32), (w, h),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        crop = f[cy0:cy1, cx0:cx1]
        restored.append(cv2.resize(crop, out_size, interpolation=cv2.INTER_AREA))
    return restored


def apply_transition(prev_tail: np.ndarray | None, seg_frames: list[np.ndarray],
                     kind: str, k: int) -> list[np.ndarray]:
    """crossfade/mask_wipe:段首 k 帧从 prev_tail 溶解过来(保持帧数)。"""
    if kind == "hard_cut" or prev_tail is None or k <= 0:
        return seg_frames
    k = min(k, len(seg_frames))
    out = list(seg_frames)
    for i in range(k):
        a = (i + 1) / (k + 1)  # 0→1
        if kind == "mask_wipe":
            # 简化的横向擦除
            mask = np.zeros(seg_frames[i].shape[:2], np.float32)
            cut = int(seg_frames[i].shape[1] * a)
            mask[:, :cut] = 1.0
            m = mask[..., None]
            out[i] = (seg_frames[i] * m + prev_tail * (1 - m)).astype(np.uint8)
        else:  # crossfade
            out[i] = cv2.addWeighted(seg_frames[i], a, prev_tail, 1 - a, 0)
    return out


def run(config_path: str | None) -> str:
    cfg = load_config(config_path)
    work_root = resolve_path(cfg, "data/work")
    out_w, out_h = get(cfg, "project.resolution", [1920, 1080])
    out_size = (int(out_w), int(out_h))
    fps = float(get(cfg, "project.fps", 30))
    overscan = float(get(cfg, "camera.overscan", 0.12))
    transition = get(cfg, "project.transition", "hard_cut")
    xfade_frames = int(get(cfg, "project.transition_frames", max(1, int(fps * 0.15))))

    comp_root = os.path.join(work_root, "comp_locked")
    if not os.path.isdir(comp_root):
        raise SystemExit("缺少 data/work/comp_locked,请先运行 S7")
    sids = sorted(int(os.path.basename(d).split("_")[1])
                  for d in glob.glob(os.path.join(comp_root, "seg_*")))

    out_dir = resolve_path(cfg, "data/output")
    video.ensure_dir(out_dir)
    silent = os.path.join(tempfile.gettempdir(), "tvg_silent.mp4")

    total = 0
    prev_tail = None
    with video.FrameWriter(silent, fps, out_size) as w:
        for sid in sids:
            seg = restore_segment_frames(work_root, sid, overscan, out_size)
            seg = apply_transition(prev_tail, seg, transition, xfade_frames)
            for f in seg:
                w.write(f)
            total += len(seg)
            prev_tail = seg[-1]
            print(f"[s8] 段 {sid}: 贴回+裁剪 {len(seg)} 帧")

    final = os.path.join(out_dir, "final.mp4")
    music = resolve_path(cfg, get(cfg, "input.music", "data/input/music.wav"))
    if os.path.isfile(music):
        try:
            video.mux_audio(silent, music, final, shortest=True)
            print(f"[s8] 已混入音乐 → {final}")
        except Exception as e:
            import shutil
            shutil.copyfile(silent, final)
            print(f"[s8][告警] 混音失败({e}),输出无声版 → {final}")
    else:
        import shutil
        shutil.copyfile(silent, final)
        print(f"[s8][告警] 找不到音乐,输出无声版 → {final}")

    try:
        os.remove(silent)
    except OSError:
        pass
    print(f"[s8] 完成:{total} 帧,{transition},{out_size[0]}x{out_size[1]} → {final}")
    return final


def main() -> None:
    ap = argparse.ArgumentParser(description="S8 贴回运镜 + 拼接出片")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
