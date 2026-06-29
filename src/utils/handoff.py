"""付费产品交接(handoff):导出标准输入 → 你在产品里处理 → 结果放 ingest → 归一化。

让任意付费产品(无需 API)也能接入统一标准件链路。详见 docs/ARTIFACTS.md。
"""
from __future__ import annotations

import glob
import json
import os
import shutil

from src import contract


def handoff_dir(root: str, step: str, sid: int) -> str:
    return os.path.join(contract.work_root(root), "handoff", step, f"seg_{sid}")


def ingest_dir(root: str, step: str, sid: int) -> str:
    return os.path.join(contract.work_root(root), "ingest", step, f"seg_{sid}")


def export_handoff(root: str, step: str, sid: int, provider: str,
                   input_dirs: dict[str, str], expect: str,
                   fps: float, size: tuple[int, int], extra_files: dict | None = None) -> str:
    """导出某步某段的标准输入与操作指引到 handoff 目录。"""
    hd = handoff_dir(root, step, sid)
    os.makedirs(hd, exist_ok=True)
    for name, src in input_dirs.items():
        dst = os.path.join(hd, f"in_{name}")
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        elif os.path.isfile(src):
            os.makedirs(dst, exist_ok=True)
            shutil.copyfile(src, os.path.join(dst, os.path.basename(src)))
    for fname, content in (extra_files or {}).items():
        with open(os.path.join(hd, fname), "w", encoding="utf-8") as f:
            f.write(content)

    ing = ingest_dir(root, step, sid)
    meta = {
        "step": step, "segment": sid, "provider": provider,
        "inputs": list(input_dirs.keys()), "expect": expect,
        "ingest_dir": ing, "fps": fps, "size": list(size), "schema": contract.SCHEMA,
    }
    with open(os.path.join(hd, "HANDOFF.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    with open(os.path.join(hd, "README.txt"), "w", encoding="utf-8") as f:
        f.write(
            f"[{step} 段 {sid}] 使用产品:{provider}\n"
            f"输入见 in_*/。{expect}\n"
            f"完成后把结果(帧序列 f00000.png… 或视频)放到:\n  {ing}\n"
            f"再在 WebUI 点该步骤的 ingest,或重新运行该步。\n")
    os.makedirs(ing, exist_ok=True)
    return hd


def ingest_ready(root: str, step: str, sid: int) -> bool:
    d = ingest_dir(root, step, sid)
    return bool(glob.glob(os.path.join(d, "f*.png")) or
                glob.glob(os.path.join(d, "*.mp4")) or
                glob.glob(os.path.join(d, "*.mov")))


def ingest_to(root: str, step: str, sid: int, out_dir: str, fps: float) -> int:
    """把 ingest 结果归一化成标准输出帧序列。支持帧序列或视频。"""
    from src.utils import video
    d = ingest_dir(root, step, sid)
    os.makedirs(out_dir, exist_ok=True)
    pngs = sorted(glob.glob(os.path.join(d, "f*.png")))
    if pngs:
        for i, p in enumerate(pngs):
            shutil.copyfile(p, os.path.join(out_dir, f"f{i:05d}.png"))
        return len(pngs)
    vids = sorted(glob.glob(os.path.join(d, "*.mp4")) + glob.glob(os.path.join(d, "*.mov")))
    if vids:
        for i, fr in enumerate(video.read_frames(vids[0])):
            video.imwrite(os.path.join(out_dir, f"f{i:05d}.png"), fr)
        return contract.frame_count(out_dir)
    return 0
