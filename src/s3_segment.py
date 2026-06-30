"""S3 抠像:在锁定域上为每帧生成主体 alpha;按人物身份(p0/p1…)输出 per-person alpha。

方法(config.segment.method):
    sam2   : SAM 2 视频分割(GPU)。用每个人物的 seed_point 作点提示 → 时序追踪 →
             每人一条 alpha 轨迹 + 并集 alpha。换装按人需要它。
    median : 时序中值背景差分(无 GPU,仅并集,合成/测试用)。

输出:
    data/work/alpha/seg_<id>/f*.png        并集 alpha(单通道 0-255)
    data/work/alpha/seg_<id>/p<k>/f*.png   每个人物的 alpha(sam2 才有)
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import tempfile

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


def _feather(m: np.ndarray, feather: int) -> np.ndarray:
    if feather > 0:
        k = feather * 2 + 1
        m = cv2.GaussianBlur(m, (k, k), 0)
    return m


# ---------------- median(无 GPU 回退)----------------

def matte_median(frames: list[np.ndarray], thresh: int, feather: int) -> list[np.ndarray]:
    stack = np.stack(frames).astype(np.float32)
    bg = np.median(stack, axis=0)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    out = []
    for f in frames:
        diff = np.linalg.norm(f.astype(np.float32) - bg, axis=2)
        m = (diff > thresh).astype(np.uint8) * 255
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, 1)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, 2)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        if n > 1:
            keep = np.where(stats[1:, cv2.CC_STAT_AREA] >= max(80, int(0.0005 * m.size)))[0] + 1
            m = np.isin(lab, keep).astype(np.uint8) * 255
        out.append(_feather(m, feather))
    return out


# ---------------- SAM2(GPU 真抠像)----------------

SAM2_MODELS = {
    "small":     ("models/sam2/sam2.1_hiera_small.pt",      "configs/sam2.1/sam2.1_hiera_s.yaml"),
    "base_plus": ("models/sam2/sam2.1_hiera_base_plus.pt",  "configs/sam2.1/sam2.1_hiera_b+.yaml"),
    "large":     ("models/sam2/sam2.1_hiera_large.pt",      "configs/sam2.1/sam2.1_hiera_l.yaml"),
}


def _sam2_predictor(cfg: dict):
    """构建 SAM2 视频预测器。优先用 segment.sam2_model(small/base_plus/large),否则用 models.sam2 路径。"""
    import torch
    from sam2.build_sam import build_sam2_video_predictor

    name = get(cfg, "segment.sam2_model")
    if name in SAM2_MODELS:
        ckpt_rel, cfg_name = SAM2_MODELS[name]
        ckpt = resolve_path(cfg, ckpt_rel)
    else:
        ckpt = resolve_path(cfg, get(cfg, "models.sam2", "models/sam2/sam2.1_hiera_small.pt"))
        cfg_name = get(cfg, "models.sam2_cfg", "configs/sam2.1/sam2.1_hiera_s.yaml")
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"SAM2 权重缺失:{ckpt}(见 README 下载)")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[s3] SAM2 模型:{name or os.path.basename(ckpt)}")
    predictor = build_sam2_video_predictor(cfg_name, ckpt, device=device)
    return predictor, device


def matte_sam2(cfg: dict, frames: list[np.ndarray], persons: list[dict],
               feather: int) -> tuple[list[np.ndarray], dict[str, list[np.ndarray]]]:
    """用各人物 seed_point 作点提示,SAM2 视频追踪 → 每人 alpha + 并集。"""
    import torch

    predictor, device = _sam2_predictor(cfg)
    h, w = frames[0].shape[:2]

    # SAM2 视频接口需要一个按整数命名的 JPEG 帧目录
    tmp = tempfile.mkdtemp(prefix="sam2_")
    try:
        for i, f in enumerate(frames):
            cv2.imwrite(os.path.join(tmp, f"{i}.jpg"), f)

        autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16) \
            if device == "cuda" else _nullctx()
        with torch.inference_mode(), autocast:
            state = predictor.init_state(video_path=tmp)
            # 支持每人多个正点(seed_points 列表)或单点(seed_point);相邻舞者用多正点比负点更稳
            valid = []
            for k, p in enumerate(persons):
                pts = p.get("seed_points") or ([p["seed_point"]] if p.get("seed_point") else [])
                if not pts:
                    continue
                predictor.add_new_points_or_box(
                    inference_state=state, frame_idx=0, obj_id=k,
                    points=np.array(pts, dtype=np.float32),
                    labels=np.ones(len(pts), dtype=np.int32))
                valid.append((k, p.get("id", f"p{k}")))

            per_obj = {k: [None] * len(frames) for k, _ in valid}
            for fidx, obj_ids, mask_logits in predictor.propagate_in_video(state):
                for j, oid in enumerate(obj_ids):
                    m = (mask_logits[j] > 0.0).squeeze().cpu().numpy().astype(np.uint8) * 255
                    if m.shape != (h, w):
                        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                    if oid in per_obj:
                        per_obj[oid][fidx] = m
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)

    per_person: dict[str, list[np.ndarray]] = {}
    union = [np.zeros((h, w), np.uint8) for _ in frames]
    for k, pid in valid:
        seq = [(_feather(m, feather) if m is not None else np.zeros((h, w), np.uint8))
               for m in per_obj[k]]
        per_person[pid] = seq
        for i, m in enumerate(seq):
            union[i] = np.maximum(union[i], m)
    return union, per_person


class _nullctx:
    def __enter__(self): return self
    def __exit__(self, *a): return False


# ---------------- 调度 ----------------

def process_segment(cfg: dict, sid: int, work_root: str, method: str) -> dict:
    frames, _ = _load_locked(work_root, sid)
    feather = int(get(cfg, "compositing.feather_px", 2))
    out_dir = os.path.join(work_root, "alpha", f"seg_{sid}")
    video.ensure_dir(out_dir)
    persons = get(cfg, "persons", []) or []
    per_person = {}

    if method == "sam2":
        try:
            union, per_person = matte_sam2(cfg, frames, persons, feather)
            used = "sam2"
        except Exception as e:
            print(f"[s3][告警] SAM2 不可用({type(e).__name__}: {e})\n[s3] 回退 median")
            union = matte_median(frames, int(get(cfg, "segment.diff_thresh", 28)), feather)
            used = "median(fallback)"
    else:
        union = matte_median(frames, int(get(cfg, "segment.diff_thresh", 28)), feather)
        used = "median"

    for i, a in enumerate(union):
        video.imwrite(os.path.join(out_dir, f"f{i:05d}.png"), a)
    for pid, seq in per_person.items():
        pd = os.path.join(out_dir, pid)
        video.ensure_dir(pd)
        for i, a in enumerate(seq):
            video.imwrite(os.path.join(pd, f"f{i:05d}.png"), a)

    cov = float(np.mean([a.mean() / 255.0 for a in union])) if union else 0.0
    who = ",".join(per_person.keys()) or "(无 per-person)"
    print(f"[s3] 段 {sid}: {len(union)} 帧 alpha → {out_dir}({used};覆盖 {cov:.1%};人物 {who})")
    return {"segment": sid, "frames": len(union), "alpha_dir": out_dir,
            "coverage": cov, "method": used, "persons": list(per_person.keys())}


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
    ap = argparse.ArgumentParser(description="S3 抠像(SAM2 / median)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--segment", type=int, default=None)
    args = ap.parse_args()
    run(args.config, args.segment)


if __name__ == "__main__":
    main()
