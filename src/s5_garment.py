"""S5 换装(可选,瓶颈环节):按段把主体换成参考图款式。

设计:段内款式固定,基于 SD1.5 + ControlNet(DWPose 姿态 + Depth)锁动作与身体结构,
参考图作外观引导(IP-Adapter)。新衣更宽松时把换装 mask 外扩,露出的旧背景由 S7 的
plate 覆盖。多主体分别处理后合并。

当前实现:换装需要 torch/diffusers + 权重 + 参考图。未就绪时本步**透传**(直接复制
锁定域帧),保证整条管线可端到端跑通;装好模型后在 `swap_frames` 里接入即可。

输入:data/work/locked/seg_{id}/   (config.enable_garment_swap=false 时也走透传)
输出:data/work/garment/seg_{id}/
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import video  # noqa: E402
from src.utils.config import get, load_config, resolve_path  # noqa: E402


def swap_available(cfg: dict) -> tuple[bool, str]:
    """检查换装所需依赖/权重是否就绪。"""
    try:
        import torch  # noqa: F401
        import diffusers  # noqa: F401
    except Exception:
        return False, "未安装 torch/diffusers"
    cn = resolve_path(cfg, get(cfg, "models.controlnet_pose", "models/controlnet-openpose"))
    if not os.path.isdir(cn):
        return False, "缺少 ControlNet 权重(models/)"
    return True, "ok"


def swap_frames(cfg, sid, locked_dir, alpha_dir, garment_refs, out_dir):  # pragma: no cover
    """真正的视频换装(待接入):SD1.5+ControlNet+IP-Adapter,逐帧/段内一致。"""
    raise NotImplementedError(
        "换装核心待接入:加载 SD1.5+ControlNet(pose/depth)+IP-Adapter(参考图),"
        "对 alpha 区域做 inpaint;固定 seed + 时序约束(AnimateDiff/光流)保证段内一致。"
    )


def _passthrough(locked_dir: str, out_dir: str) -> int:
    video.ensure_dir(out_dir)
    paths = sorted(glob.glob(os.path.join(locked_dir, "f*.png")))
    for p in paths:
        shutil.copyfile(p, os.path.join(out_dir, os.path.basename(p)))
    return len(paths)


def process_segment(cfg: dict, sid: int, work_root: str, enabled: bool) -> dict:
    locked_dir = os.path.join(work_root, "locked", f"seg_{sid}")
    out_dir = os.path.join(work_root, "garment", f"seg_{sid}")
    mode = "passthrough"

    if enabled:
        ok, why = swap_available(cfg)
        if ok:
            try:
                alpha_dir = os.path.join(work_root, "alpha", f"seg_{sid}")
                swap_frames(cfg, sid, locked_dir, alpha_dir, None, out_dir)
                mode = "swap"
            except NotImplementedError as e:
                print(f"[s5][告警] {e}\n[s5] 段 {sid} 透传")
                _passthrough(locked_dir, out_dir)
        else:
            print(f"[s5][告警] 换装不可用({why}),段 {sid} 透传")
            _passthrough(locked_dir, out_dir)
    else:
        _passthrough(locked_dir, out_dir)

    n = len(glob.glob(os.path.join(out_dir, "f*.png")))
    print(f"[s5] 段 {sid}: {n} 帧 → {out_dir}({mode})")
    return {"segment": sid, "frames": n, "garment_dir": out_dir, "mode": mode}


def run(config_path: str | None, only_segment: int | None) -> list[dict]:
    cfg = load_config(config_path)
    work_root = resolve_path(cfg, "data/work")
    enabled = bool(get(cfg, "project.enable_garment_swap", True))
    locked_root = os.path.join(work_root, "locked")
    if not os.path.isdir(locked_root):
        raise SystemExit("缺少 data/work/locked,请先运行 S2")
    sids = sorted(int(os.path.basename(d).split("_")[1])
                  for d in glob.glob(os.path.join(locked_root, "seg_*")))
    if only_segment is not None:
        sids = [s for s in sids if s == only_segment]
    results = [process_segment(cfg, s, work_root, enabled) for s in sids]
    print(f"[s5] 完成 {len(results)} 段(enable_garment_swap={enabled})")
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="S5 换装(可选)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--segment", type=int, default=None)
    args = ap.parse_args()
    run(args.config, args.segment)


if __name__ == "__main__":
    main()
