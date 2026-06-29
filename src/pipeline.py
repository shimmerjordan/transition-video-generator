"""转场视频生成器 — 全流程编排。

串联 S1→S8。支持单段 POC(--segment N:S2–S7 只处理该段,S1/S8 仍全局),
以及只跑部分步骤(--steps 2-7 或 3,5,7)。

用法:
    python src/pipeline.py                  # 全流程,所有段
    python src/pipeline.py --segment 3      # 单段 POC(推荐先跑通这个)
    python src/pipeline.py --steps 2-7      # 只跑 S2..S7
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import (  # noqa: E402
    s1_beats, s2_camera, s3_segment, s4_plates,
    s5_garment, s6_relight, s7_composite, s8_restore_assemble,
)


def parse_steps(spec: str | None) -> set[int]:
    if not spec:
        return set(range(1, 9))
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        elif part:
            out.add(int(part))
    return out


def run(config_path: str | None, segment: int | None, steps: set[int]) -> None:
    t0 = time.time()
    seg = segment  # None = 全部
    print(f"=== 转场视频生成器:steps={sorted(steps)}"
          f"{f',段 {segment}' if segment is not None else ',全部段'} ===")

    if 1 in steps:
        s1_beats.run(config_path, None, False)
    if 2 in steps:
        s2_camera.run(config_path, seg)
    if 3 in steps:
        s3_segment.run(config_path, seg)
    if 4 in steps:
        s4_plates.run(config_path, seg)
    if 5 in steps:
        s5_garment.run(config_path, seg)
    if 6 in steps:
        s6_relight.run(config_path, seg)
    if 7 in steps:
        s7_composite.run(config_path, seg)
    if 8 in steps:
        s8_restore_assemble.run(config_path)

    print(f"=== 完成,用时 {time.time() - t0:.1f}s ===")


def main() -> None:
    ap = argparse.ArgumentParser(description="转场视频生成器全流程")
    ap.add_argument("--config", default=None)
    ap.add_argument("--segment", type=int, default=None, help="单段 POC(S2–S7 仅该段)")
    ap.add_argument("--steps", default=None, help="如 2-7 或 3,5,7;默认全部")
    args = ap.parse_args()
    run(args.config, args.segment, parse_steps(args.steps))


if __name__ == "__main__":
    main()
