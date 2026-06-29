"""转场视频生成器 — 全流程编排(provider 感知)。

每步按 config.providers 选 本地 or 现成产品(product:* 走 handoff),统一标准件 I/O。
步骤顺序与编号:
    1 beats  2 camera  3 matte  4 plates  5 garment  6 relight  7 composite  8 assemble

用法:
    python src/pipeline.py                  # 全流程,所有段
    python src/pipeline.py --segment 0      # 单段 POC
    python src/pipeline.py --steps 3,5      # 只跑 matte、garment
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import providers  # noqa: E402
from src.utils.config import load_config, project_root  # noqa: E402

STEP_ORDER = ["beats", "camera", "matte", "plates", "garment", "relight", "composite", "assemble"]


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


def run(config_path: str | None, segment: int | None, steps: set[int],
        log=print) -> None:
    t0 = time.time()
    cfg = load_config(config_path)
    root = project_root()
    sel = [STEP_ORDER[i - 1] for i in sorted(steps) if 1 <= i <= 8]
    log(f"=== 转场视频生成器:{sel}"
        f"{f',段 {segment}' if segment is not None else ',全部段'} ===")
    for step in sel:
        prov = providers.resolve(cfg, step)
        log(f"--- {step}(provider={prov})---")
        for line in providers.run_step(step, config_path, cfg, root, segment):
            log(line)
    log(f"=== 完成,用时 {time.time() - t0:.1f}s ===")


def main() -> None:
    ap = argparse.ArgumentParser(description="转场视频生成器全流程")
    ap.add_argument("--config", default=None)
    ap.add_argument("--segment", type=int, default=None, help="单段 POC")
    ap.add_argument("--steps", default=None, help="如 2-7 或 3,5;默认全部")
    args = ap.parse_args()
    run(args.config, args.segment, parse_steps(args.steps))


if __name__ == "__main__":
    main()
