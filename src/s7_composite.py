"""S7 合成(锁定域):把主体叠到背景平面上,并做"像实拍"的真实度处理。

处理项(均可在 config.compositing 开关):
    color match  : 主体色统计向背景温和靠拢,消除色调割裂
    light wrap   : 背景边缘光"溢"到主体轮廓,消除剪贴感
    contact shadow: 主体下方投软阴影,增加接地感
    grain        : 全画面统一胶片颗粒
    bg blur      : 背景轻微虚化,模拟景深(可选)

输入:data/work/relit/seg_{id}/ + alpha/seg_{id}/ + plates/seg_{id}/
输出:data/work/comp_locked/seg_{id}/f{idx:05d}.png
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


def defringe(subject: np.ndarray, alpha: np.ndarray, band: int,
             core_thr: int = 200) -> np.ndarray:
    """边缘去污染:半透明边缘带的 RGB 仍带着原视频的旧背景色,叠到新背景会形成虚影/光晕。
    把边缘带(实心前景之外的一圈)用实心前景色外扩重绘,再用软 alpha 混合 → 边缘只剩人物色渐隐。"""
    if band <= 0:
        return subject
    core = (alpha >= core_thr).astype(np.uint8)
    if int(core.sum()) < 10:
        return subject
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (band * 2 + 1, band * 2 + 1))
    grown = cv2.dilate(core, k)
    fill = (((grown > 0) & (core == 0)).astype(np.uint8)) * 255   # 待重绘的边缘环
    if int(fill.sum()) == 0:
        return subject
    # inpaint 只用未遮罩区域(=实心前景)外推,从而把前景色"长"进边缘环,挤掉旧背景色
    return cv2.inpaint(subject, fill, 3, cv2.INPAINT_TELEA)


def color_match(subject: np.ndarray, plate: np.ndarray, alpha: np.ndarray,
                strength: float) -> np.ndarray:
    """把主体区域的均值向背景均值靠拢(温和)。"""
    a = (alpha > 16)
    if a.sum() < 10:
        return subject
    s = subject.astype(np.float32)
    smean = s[a].mean(0)
    pmean = plate.reshape(-1, 3).mean(0)
    shift = (pmean - smean) * strength
    out = s + shift * (alpha.astype(np.float32) / 255.0)[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def light_wrap(subject: np.ndarray, plate: np.ndarray, alpha: np.ndarray,
               amount: float, width: int) -> np.ndarray:
    """背景模糊色按内缘带混入主体,形成光包裹。"""
    k = max(3, width * 2 + 1)
    inner = cv2.erode(alpha, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    rim = cv2.GaussianBlur(cv2.subtract(alpha, inner), (k, k), 0).astype(np.float32) / 255.0
    rim = (rim * amount)[..., None]
    plate_blur = cv2.GaussianBlur(plate, (k * 2 + 1, k * 2 + 1), 0).astype(np.float32)
    out = subject.astype(np.float32) * (1 - rim) + plate_blur * rim
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_contact_shadow(plate: np.ndarray, alpha: np.ndarray,
                         dy: int, blur: int, strength: float) -> np.ndarray:
    """在背景上、主体下方投软阴影(贴回前先压暗背景)。"""
    M = np.float32([[1, 0, 0], [0, 1, dy]])
    sh = cv2.warpAffine(alpha, M, (alpha.shape[1], alpha.shape[0]))
    kb = blur * 2 + 1
    sh = cv2.GaussianBlur(sh, (kb, kb), 0).astype(np.float32) / 255.0
    sh = (sh * strength)[..., None]
    return np.clip(plate.astype(np.float32) * (1 - sh), 0, 255).astype(np.uint8)


def add_grain(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return img
    noise = np.random.normal(0, sigma, img.shape[:2]).astype(np.float32)[..., None]
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def composite_frame(subject, plate, alpha, cfg) -> np.ndarray:
    c = get(cfg, "compositing", {}) or {}
    # 边缘:先轻微收边(把软过渡带拉进人物内,削掉旧背景外缘)
    ee = int(c.get("edge_erode", 1))
    if ee > 0:
        alpha = cv2.erode(alpha, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ee * 2 + 1, ee * 2 + 1)))
    # 收紧半透明带:把 0/255 之间的灰边对比度拉高,减少旧背景透出形成的虚影外圈
    et = float(c.get("edge_tighten", 0.15))
    if et > 0:
        lo, hi = et, 1.0 - et
        a = (alpha.astype(np.float32) / 255.0 - lo) / max(1e-3, hi - lo)
        alpha = np.clip(a, 0, 1)
        alpha = (alpha * 255.0).astype(np.uint8)
    # 去污染:用实心前景色外扩重绘边缘带,挤掉旧背景色(band 需覆盖软边宽度)
    if c.get("defringe", True):
        subject = defringe(subject, alpha, int(c.get("defringe_band", 8)))
    if c.get("bg_blur", 0):
        kb = int(c["bg_blur"]) * 2 + 1
        plate = cv2.GaussianBlur(plate, (kb, kb), 0)
    if c.get("contact_shadow", True):
        plate = apply_contact_shadow(
            plate, alpha,
            dy=int(c.get("shadow_dy", max(4, plate.shape[0] // 120))),
            blur=int(c.get("shadow_blur", 9)),
            strength=float(c.get("shadow_strength", 0.45)))
    if c.get("match_color", True):
        subject = color_match(subject, plate, alpha, float(c.get("match_strength", 0.25)))
    if c.get("light_wrap", True):
        subject = light_wrap(subject, plate, alpha,
                             amount=float(c.get("light_wrap_amount", 0.5)),
                             width=int(c.get("light_wrap_width", 4)))
    a = (alpha.astype(np.float32) / 255.0)[..., None]
    out = subject.astype(np.float32) * a + plate.astype(np.float32) * (1 - a)
    out = out.astype(np.uint8)
    if c.get("grain", True):
        out = add_grain(out, float(c.get("grain_sigma", 3.0)))
    return out


def process_segment(cfg: dict, sid: int, work_root: str) -> dict:
    alpha_dir = os.path.join(work_root, "alpha", f"seg_{sid}")
    plate_dir = os.path.join(work_root, "plates", f"seg_{sid}")
    out_dir = os.path.join(work_root, "comp_locked", f"seg_{sid}")
    video.ensure_dir(out_dir)

    # 主体层来源:重打光 > 换装 > 锁定原片(只换背景时回退原片人物)
    subject_dir = subject_kind = None
    for cand in ("relit", "garment", "locked"):
        d = os.path.join(work_root, cand, f"seg_{sid}")
        if glob.glob(os.path.join(d, "f*.png")):
            subject_dir, subject_kind = d, cand
            break
    if subject_dir is None:
        raise SystemExit(f"段 {sid} 缺少主体帧(locked/garment/relit 都没有),请先运行 S2/S3")
    print(f"[s7] 段 {sid}: 主体层用 {subject_kind}")
    names = [os.path.basename(p) for p in sorted(glob.glob(os.path.join(subject_dir, "f*.png")))]

    for name in names:
        subject = video.imread(os.path.join(subject_dir, name))
        alpha = video.imread(os.path.join(alpha_dir, name), cv2.IMREAD_GRAYSCALE)
        plate = video.imread(os.path.join(plate_dir, name))
        if alpha is None:
            alpha = np.full(subject.shape[:2], 255, np.uint8)
        if plate is None:
            plate = np.zeros_like(subject)
        if plate.shape[:2] != subject.shape[:2]:
            plate = cv2.resize(plate, (subject.shape[1], subject.shape[0]))
        out = composite_frame(subject, plate, alpha, cfg)
        video.imwrite(os.path.join(out_dir, name), out)

    print(f"[s7] 段 {sid}: {len(names)} 帧合成 → {out_dir}")
    return {"segment": sid, "frames": len(names), "comp_dir": out_dir}


def run(config_path: str | None, only_segment: int | None) -> list[dict]:
    cfg = load_config(config_path)
    work_root = resolve_path(cfg, "data/work")
    # 段集合取自 alpha(抠像已出)即可合成;主体层在 process_segment 里回退
    base = os.path.join(work_root, "alpha")
    if not os.path.isdir(base):
        raise SystemExit("缺少 data/work/alpha,请先运行抠像(S3)")
    sids = sorted(int(os.path.basename(d).split("_")[1])
                  for d in glob.glob(os.path.join(base, "seg_*")))
    if only_segment is not None:
        sids = [s for s in sids if s == only_segment]
    results = [process_segment(cfg, s, work_root) for s in sids]
    print(f"[s7] 完成 {len(results)} 段")
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="S7 合成 + 真实度处理")
    ap.add_argument("--config", default=None)
    ap.add_argument("--segment", type=int, default=None)
    args = ap.parse_args()
    run(args.config, args.segment)


if __name__ == "__main__":
    main()
