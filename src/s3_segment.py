"""S3 抠像:在锁定域上为每帧生成主体 alpha matte。

两种方法(config.segment.method):
    median : 时序中值背景差分(默认,无需 GPU/模型)。因 S2 已把画面锁定,
             背景近似静止,对多帧取中值即得干净背景,差分即得运动主体——
             这是离线即可用的真实抠像,适合先把整条管线跑通。
    sam2   : SAM 2 视频分割(质量升级,需模型权重 + 提示点)。当前为占位,
             选中而不可用时回退 median。

输出:data/work/alpha/seg_{id}/f{idx:05d}.png(单通道 0-255 alpha)
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import video  # noqa: E402
from src.utils.config import get, load_config, resolve_path  # noqa: E402


def _load_locked(work_root: str, sid: int) -> tuple[list[np.ndarray], list[str]]:
    d = os.path.join(work_root, "locked", f"seg_{sid}")
    paths = sorted(glob.glob(os.path.join(d, "f*.png")))
    if not paths:
        raise SystemExit(f"段 {sid} 缺少锁定域帧({d}),请先运行 S2")
    return [video.imread(p) for p in paths], paths


def matte_median(frames: list[np.ndarray], thresh: int, feather: int) -> list[np.ndarray]:
    """时序中值背景差分 → 主体 alpha。"""
    stack = np.stack(frames).astype(np.float32)
    bg = np.median(stack, axis=0)  # HxWx3 背景估计
    alphas = []
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    for f in frames:
        diff = np.linalg.norm(f.astype(np.float32) - bg, axis=2)  # 颜色距离
        m = (diff > thresh).astype(np.uint8) * 255
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
        # 取最大若干连通域,去噪点
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        if n > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            keep = np.where(areas >= max(80, int(0.0005 * m.size)))[0] + 1
            m = np.isin(lab, keep).astype(np.uint8) * 255
        if feather > 0:
            kf = feather * 2 + 1
            m = cv2.GaussianBlur(m, (kf, kf), 0)
        alphas.append(m)
    return alphas


def matte_sam2(frames, model_dir):  # pragma: no cover - 占位
    raise NotImplementedError(
        "SAM2 抠像尚未接入。请安装 sam2 与权重后在此实现视频分割;"
        "当前请使用 config.segment.method=median。"
    )


def process_segment(cfg: dict, sid: int, work_root: str, method: str) -> dict:
    frames, _ = _load_locked(work_root, sid)
    thresh = int(get(cfg, "segment.diff_thresh", 28))
    feather = int(get(cfg, "compositing.feather_px", 2))

    if method == "sam2":
        try:
            alphas = matte_sam2(frames, resolve_path(cfg, get(cfg, "models.sam2", "models/sam2")))
        except NotImplementedError as e:
            print(f"[s3][告警] {e}\n[s3] 回退 median")
            alphas = matte_median(frames, thresh, feather)
    else:
        alphas = matte_median(frames, thresh, feather)

    out_dir = os.path.join(work_root, "alpha", f"seg_{sid}")
    video.ensure_dir(out_dir)
    for i, a in enumerate(alphas):
        video.imwrite(os.path.join(out_dir, f"f{i:05d}.png"), a)
    cov = float(np.mean([a.mean() / 255.0 for a in alphas]))
    print(f"[s3] 段 {sid}: {len(alphas)} 帧 alpha → {out_dir}(平均覆盖 {cov:.1%})")
    return {"segment": sid, "frames": len(alphas), "alpha_dir": out_dir, "coverage": cov}


def run(config_path: str | None, only_segment: int | None) -> list[dict]:
    cfg = load_config(config_path)
    work_root = resolve_path(cfg, "data/work")
    method = get(cfg, "segment.method", "median")
    locked_root = os.path.join(work_root, "locked")
    if not os.path.isdir(locked_root):
        raise SystemExit("缺少 data/work/locked,请先运行 S2")
    sids = sorted(int(os.path.basename(d).split("_")[1])
                  for d in glob.glob(os.path.join(locked_root, "seg_*")))
    if only_segment is not None:
        sids = [s for s in sids if s == only_segment]
    results = [process_segment(cfg, s, work_root, method) for s in sids]
    print(f"[s3] 完成 {len(results)} 段(method={method})")
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="S3 抠像")
    ap.add_argument("--config", default=None)
    ap.add_argument("--segment", type=int, default=None)
    args = ap.parse_args()
    run(args.config, args.segment)


if __name__ == "__main__":
    main()
