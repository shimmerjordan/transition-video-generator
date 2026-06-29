"""S5 换装(可选,按人按段):依据 config.segments[*].garments(person→garment)换装。

- 服装库:config.garments(id→图片);人物身份:config.persons(p0/p1…,来自 matte 轨迹)。
- provider=local:目前无本地试衣模型 → 透传(复制锁定域帧),并写出 swap_plan.json 记录
  "本段哪个人换哪张参考图",供 WebUI 展示与 product 交接使用。
- provider=product:* 由 providers.run_product 走 handoff(自动附带服装参考图与 swap_plan)。
  接入本地 SD+ControlNet+IP-Adapter 后,在 swap_frames 实现即可。

输入:locked + alpha(+ per-person alpha)   输出:garment
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import contract  # noqa: E402
from src.utils import video  # noqa: E402
from src.utils.config import get, load_config, resolve_path  # noqa: E402


def _seg_cfg(cfg: dict, sid: int) -> dict:
    for s in get(cfg, "segments", []) or []:
        if s.get("id") == sid:
            return s
    return {"id": sid}


def resolve_swap_plan(cfg: dict, sid: int) -> dict:
    """返回 {person_id: {"name":.., "garment":id, "image":abs_path}}。"""
    g_dir = resolve_path(cfg, get(cfg, "input.garments_dir", "data/input/garments"))
    lib = get(cfg, "garments", {}) or {}
    names = {p["id"]: p.get("name", p["id"]) for p in get(cfg, "persons", []) or []}
    plan = {}
    for person, gid in (_seg_cfg(cfg, sid).get("garments", {}) or {}).items():
        img = lib.get(gid)
        if not img:
            print(f"[s5] 段 {sid}: 服装库无 '{gid}',跳过 {person}")
            continue
        plan[person] = {"name": names.get(person, person), "garment": gid,
                        "image": os.path.join(g_dir, os.path.basename(img))}
    return plan


def swap_available(cfg: dict) -> tuple[bool, str]:
    try:
        import torch  # noqa: F401
        import diffusers  # noqa: F401
    except Exception:
        return False, "未安装 torch/diffusers"
    if not os.path.isdir(resolve_path(cfg, get(cfg, "models.controlnet_pose", "models/controlnet-openpose"))):
        return False, "缺少 ControlNet 权重"
    return True, "ok"


def swap_frames(cfg, sid, locked_dir, alpha_dir, plan, out_dir):  # pragma: no cover
    raise NotImplementedError(
        "本地换装待接入:SD1.5+ControlNet(pose/depth)+IP-Adapter,按 per-person alpha 分别 inpaint。")


def _passthrough(locked_dir: str, out_dir: str) -> int:
    video.ensure_dir(out_dir)
    for p in sorted(glob.glob(os.path.join(locked_dir, "f*.png"))):
        shutil.copyfile(p, os.path.join(out_dir, os.path.basename(p)))
    return contract.frame_count(out_dir)


def process_segment(cfg: dict, sid: int, work_root: str) -> dict:
    locked_dir = os.path.join(work_root, "locked", f"seg_{sid}")
    out_dir = os.path.join(work_root, "garment", f"seg_{sid}")
    plan = resolve_swap_plan(cfg, sid)

    video.ensure_dir(out_dir)
    with open(os.path.join(out_dir, "swap_plan.json"), "w", encoding="utf-8") as f:
        json.dump({"segment": sid, "swaps": plan}, f, ensure_ascii=False, indent=2)

    mode = "passthrough"
    if plan:
        ok, why = swap_available(cfg)
        if ok:
            try:
                swap_frames(cfg, sid, locked_dir, os.path.join(work_root, "alpha", f"seg_{sid}"), plan, out_dir)
                mode = "swap"
            except NotImplementedError as e:
                print(f"[s5][告警] {e}\n[s5] 段 {sid} 透传(本地无模型;可改 provider=product)")
                _passthrough(locked_dir, out_dir)
        else:
            print(f"[s5][告警] 本地换装不可用({why}),段 {sid} 透传;建议 provider=product")
            _passthrough(locked_dir, out_dir)
    else:
        _passthrough(locked_dir, out_dir)

    n = contract.frame_count(out_dir)
    who = ", ".join(f"{v['name']}→{v['garment']}" for v in plan.values()) or "无"
    print(f"[s5] 段 {sid}: {n} 帧 → {out_dir}({mode};换装计划:{who})")
    return {"segment": sid, "frames": n, "garment_dir": out_dir, "mode": mode, "plan": plan}


def run(config_path: str | None, only_segment: int | None) -> list[dict]:
    cfg = load_config(config_path)
    work_root = resolve_path(cfg, "data/work")
    locked_root = os.path.join(work_root, "locked")
    if not os.path.isdir(locked_root):
        raise SystemExit("缺少 data/work/locked,请先运行 S2")
    sids = sorted(int(os.path.basename(d).split("_")[1])
                  for d in glob.glob(os.path.join(locked_root, "seg_*")))
    if only_segment is not None:
        sids = [s for s in sids if s == only_segment]
    results = [process_segment(cfg, s, work_root) for s in sids]
    print(f"[s5] 完成 {len(results)} 段")
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="S5 换装(按人按段)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--segment", type=int, default=None)
    args = ap.parse_args()
    run(args.config, args.segment)


if __name__ == "__main__":
    main()
