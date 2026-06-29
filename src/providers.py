"""Provider 解析:每个步骤选 本地 or 现成产品(统一标准件 I/O)。

- local      → 调用步骤模块的 run()(直接产出标准件)。
- product:*  → 走 handoff:若 ingest 已就绪则归一化为标准输出;否则导出 handoff 包并提示。

camera/composite/assemble 固定本地(几何与封装,不外包)。
"""
from __future__ import annotations

import importlib
import os

from src import contract
from src.utils import handoff
from src.utils.config import get


def resolve(cfg: dict, step: str) -> str:
    """返回该步骤的 provider 字符串,如 'local' 或 'product:kling'。"""
    return get(cfg, f"providers.{step}", "local")


def is_product(provider: str) -> tuple[bool, str]:
    if provider.startswith("product:"):
        return True, provider.split(":", 1)[1]
    return False, ""


def run_local(step: str, config_path: str | None, only_segment: int | None):
    """调用步骤模块的 run()(本地实现)。"""
    spec = contract.STEP_SPECS[step]
    mod = importlib.import_module(f"src.{spec['module']}")
    if step in ("beats", "assemble"):       # 非分段或全局步骤
        if step == "beats":
            return mod.run(config_path, None, False)
        return mod.run(config_path)
    return mod.run(config_path, only_segment)


def run_product(step: str, cfg: dict, root: str, sid: int, provider: str) -> str:
    """付费产品路径:ingest 就绪→归一化;否则导出 handoff 包。返回状态串。"""
    spec = contract.STEP_SPECS[step]
    fps = float(get(cfg, "project.fps", 30))
    w, h = get(cfg, "project.resolution", [1280, 720])

    if handoff.ingest_ready(root, step, sid):
        out_dir = contract.seg_dir(root, spec["output"], sid)
        n = handoff.ingest_to(root, step, sid, out_dir, fps)
        contract.write_manifest(out_dir, step=step, segment=sid, kind=spec["kind"],
                                fps=fps, width=int(w), height=int(h), count=n,
                                provider=provider)
        return f"ingested {n} 帧 → {out_dir}"

    # 导出标准输入
    input_dirs = {name: contract.seg_dir(root, name, sid) for name in spec["inputs"]}
    extra_files = None
    # garment:附带服装参考图与 swap 计划
    if step == "garment":
        import json as _json
        from src import s5_garment
        plan = s5_garment.resolve_swap_plan(cfg, sid)
        refs_dir = os.path.join(handoff.handoff_dir(root, step, sid), "in_garment_refs")
        os.makedirs(refs_dir, exist_ok=True)
        import shutil as _sh
        for person, v in plan.items():
            if os.path.isfile(v["image"]):
                _sh.copyfile(v["image"], os.path.join(refs_dir, f"{person}_{os.path.basename(v['image'])}"))
        extra_files = {"swap_plan.json": _json.dumps({"segment": sid, "swaps": plan}, ensure_ascii=False, indent=2)}
    expect = {
        "matte": "对每帧把主体抠出,导出灰度 alpha 帧序列(主体白、背景黑)。",
        "garment": "把参考服装换到主体身上,导出换装后的主体帧序列(锁定域,尺寸不变)。",
        "cleanup": "去除水印/字幕/路人,导出干净背景帧序列。",
        "ground": "为画面补出真实地面/场景,导出帧序列。",
        "relight": "按背景光照对主体重打光,导出帧序列。",
    }.get(step, "处理后导出标准帧序列。")
    hd = handoff.export_handoff(root, step, sid, provider, input_dirs, expect,
                               fps, (int(w), int(h)), extra_files=extra_files)
    return f"已导出 handoff 包 → {hd}(用 {provider} 处理后把结果放 ingest 再运行)"


def run_step(step: str, config_path: str | None, cfg: dict, root: str,
             only_segment: int | None) -> list[str]:
    """统一入口:按 provider 分派 local / product。"""
    provider = resolve(cfg, step)
    prod, name = is_product(provider)
    if not prod:
        run_local(step, config_path, only_segment)
        return [f"[{step}] local 完成"]

    # product:按段处理
    if step in ("beats", "assemble"):
        run_local(step, config_path, only_segment)  # 这两步不外包
        return [f"[{step}] local(不外包)"]
    segs = _segment_ids(cfg, root, only_segment)
    return [f"[{step} 段 {sid}] " + run_product(step, cfg, root, sid, provider) for sid in segs]


def _segment_ids(cfg: dict, root: str, only_segment: int | None) -> list[int]:
    if only_segment is not None:
        return [only_segment]
    import glob
    locked = os.path.join(contract.work_root(root), "locked")
    ids = sorted(int(os.path.basename(d).split("_")[1])
                 for d in glob.glob(os.path.join(locked, "seg_*")))
    if ids:
        return ids
    return [s["id"] for s in get(cfg, "segments", []) or []]
