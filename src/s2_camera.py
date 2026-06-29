"""S2 运镜跟踪与反稳定:把(可能带运镜的)源片每帧的摄像机运动 T_t 估出来,
并反稳定到"锁定域",供 S3–S7 在静止画面里处理;T_t 存盘供 S8 贴回。

策略(motion_2d,默认):逐帧光流跟踪 → 累积仿射 A_t(参考帧→第 t 帧的相机运动)
→ 反稳定帧 L_t = warp(frame_t, A_t⁻¹)。S8 再以 A_t 把运镜贴回。

相机模式(config.camera.mode):
    locked    : 固定机位,A_t 全为单位阵,直接用原帧
    motion_2d : 上述 2D 反贴(覆盖摇/俯仰/滚转/变焦)
    motion_3d : 强视差时的三维解算(本实现暂回退到 motion_2d 并告警)
    auto      : 估计后按位移幅度自动判定 locked / motion_2d

输出:
    data/work/locked/seg_{id}/f{idx:05d}.png   反稳定后的锁定域帧
    data/work/camera/seg_{id}.json             A_t 序列(2x3)+ meta

用法:
    python src/s2_camera.py                 # 处理所有段
    python src/s2_camera.py --segment 3     # 只处理第 3 段(POC)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import video  # noqa: E402
from src.utils.config import get, load_config, resolve_path  # noqa: E402

IDENTITY = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)


def _to3x3(a: np.ndarray) -> np.ndarray:
    m = np.eye(3, dtype=np.float64)
    m[:2] = a
    return m


def estimate_affine(prev_gray: np.ndarray, cur_gray: np.ndarray,
                    mask: np.ndarray | None = None) -> np.ndarray:
    """估计 prev→cur 的相似变换(平移+旋转+缩放),失败回退单位阵。"""
    p0 = cv2.goodFeaturesToTrack(prev_gray, maxCorners=2000, qualityLevel=0.01,
                                 minDistance=8, mask=mask)
    if p0 is None or len(p0) < 12:
        return IDENTITY.copy()
    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, cur_gray, p0, None)
    if p1 is None:
        return IDENTITY.copy()
    st = st.reshape(-1).astype(bool)
    g0, g1 = p0[st], p1[st]
    if len(g0) < 12:
        return IDENTITY.copy()
    M, inliers = cv2.estimateAffinePartial2D(g0, g1, method=cv2.RANSAC,
                                             ransacReprojThreshold=3.0)
    if M is None:
        return IDENTITY.copy()
    return M.astype(np.float64)


def track_segment(frames: list[np.ndarray],
                  subject_masks: list[np.ndarray] | None = None) -> list[np.ndarray]:
    """返回每帧 A_t(参考帧=第0帧 → 第t帧)的 2x3 仿射列表。"""
    transforms = [IDENTITY.copy()]
    if not frames:
        return transforms
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    acc = _to3x3(IDENTITY)
    for i in range(1, len(frames)):
        cur_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        # 用主体 mask 排除舞者区域(255=可用作跟踪的背景)
        track_mask = None
        if subject_masks is not None and i - 1 < len(subject_masks):
            m = subject_masks[i - 1]
            track_mask = np.where(m > 127, np.uint8(0), np.uint8(255))
        M = estimate_affine(prev_gray, cur_gray, track_mask)
        acc = _to3x3(M) @ acc
        transforms.append(acc[:2].copy())
        prev_gray = cur_gray
    return transforms


def motion_magnitude(transforms: list[np.ndarray], size: tuple[int, int]) -> float:
    """累积运镜的最大角点位移(像素),用于 auto 判定。"""
    w, h = size
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)
    max_disp = 0.0
    for A in transforms:
        warped = (A[:, :2] @ corners.T).T + A[:, 2]
        max_disp = max(max_disp, float(np.abs(warped - corners).max()))
    return max_disp


def stabilize(frames: list[np.ndarray], transforms: list[np.ndarray]) -> list[np.ndarray]:
    """L_t = warp(frame_t, A_t⁻¹),把每帧拉回参考视图(锁定域)。"""
    h, w = frames[0].shape[:2]
    out = []
    for f, A in zip(frames, transforms):
        inv = cv2.invertAffineTransform(A)
        out.append(cv2.warpAffine(f, inv, (w, h), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REFLECT))
    return out


def process_segment(cfg: dict, seg: dict, source_path: str, work_root: str,
                    mode: str) -> dict:
    sid = seg["id"]
    start = seg["start_frame"]
    count = max(1, seg["end_frame"] - seg["start_frame"])
    frames = list(video.read_frames(source_path, start=start, count=count))
    if not frames:
        raise SystemExit(f"段 {sid} 未读到帧(start={start})")
    h, w = frames[0].shape[:2]

    if mode == "locked":
        transforms = [IDENTITY.copy() for _ in frames]
        eff_mode = "locked"
    else:
        if mode == "motion_3d":
            print(f"[s2][告警] motion_3d 尚未实现三维解算,段 {sid} 回退 motion_2d")
        transforms = track_segment(frames)
        if mode == "auto":
            disp = motion_magnitude(transforms, (w, h))
            eff_mode = "locked" if disp < 2.0 else "motion_2d"
            print(f"[s2] 段 {sid} 运镜最大位移 {disp:.1f}px → {eff_mode}")
            if eff_mode == "locked":
                transforms = [IDENTITY.copy() for _ in frames]
        else:
            eff_mode = "motion_2d"

    locked = (frames if eff_mode == "locked"
              else stabilize(frames, transforms))

    # 写锁定域帧
    locked_dir = os.path.join(work_root, "locked", f"seg_{sid}")
    video.ensure_dir(locked_dir)
    for i, f in enumerate(locked):
        video.imwrite(os.path.join(locked_dir, f"f{i:05d}.png"), f)

    # 写 T_t
    cam_path = os.path.join(work_root, "camera", f"seg_{sid}.json")
    video.save_transforms(cam_path, transforms,
                          meta={"segment": sid, "mode": eff_mode,
                                "width": w, "height": h, "frames": len(frames)})
    print(f"[s2] 段 {sid}: {len(frames)} 帧 → {locked_dir}({eff_mode})")
    return {"segment": sid, "frames": len(frames), "mode": eff_mode,
            "locked_dir": locked_dir, "camera": cam_path}


def run(config_path: str | None, only_segment: int | None) -> list[dict]:
    cfg = load_config(config_path)
    source_path = resolve_path(cfg, get(cfg, "input.source", "data/input/source.mp4"))
    if not os.path.isfile(source_path):
        raise SystemExit(f"找不到源视频:{source_path}(请放入 data/input/)")

    beats_path = resolve_path(cfg, "data/work/beats.json")
    if not os.path.isfile(beats_path):
        raise SystemExit("缺少 data/work/beats.json,请先运行 S1")
    beats = json.load(open(beats_path, encoding="utf-8"))

    mode = get(cfg, "camera.mode", "auto")
    work_root = resolve_path(cfg, "data/work")
    segs = beats["segments"]
    if only_segment is not None:
        segs = [s for s in segs if s["id"] == only_segment]
        if not segs:
            raise SystemExit(f"beats.json 中没有段 {only_segment}")

    results = [process_segment(cfg, s, source_path, work_root, mode) for s in segs]
    print(f"[s2] 完成 {len(results)} 段(mode={mode})")
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="S2 运镜跟踪与反稳定")
    ap.add_argument("--config", default=None)
    ap.add_argument("--segment", type=int, default=None, help="只处理某一段(POC)")
    args = ap.parse_args()
    run(args.config, args.segment)


if __name__ == "__main__":
    main()
